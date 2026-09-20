"""Phase 13 scheduled regression pipeline (Airflow DAG + dry-run share this).

Steps: resolve prompt config → eval harness → Evidently suite → one MLflow run
→ compare vs last promoted baseline → alert on degradation.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from app.agent.prompts import research_prompt_path
from app.config import settings
from eval.harness import DEFAULT_REPORT, run_harness
from eval.scoring import AggregateMetrics
from mlops.evidently_regression import RegressionResult, run_regression
from mlops.tracking import (
    configure_mlflow,
    get_last_promoted_metrics,
    log_evidently_metrics,
    log_evidently_report,
    log_harness_metrics,
    log_prompt_and_traces,
    start_experiment_run,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALERT_PATH = REPO_ROOT / "mlops" / "reports" / "regression_alert.md"
DEFAULT_SCHEDULE = "0 2 * * *"


@dataclass
class PipelineConfig:
    prompt_version: str
    degrade_pp: float
    evidently_threshold: float
    tracking_uri: str
    experiment_name: str
    webhook_url: str
    schedule: str = DEFAULT_SCHEDULE
    prompt_file: Path = field(default_factory=Path)


@dataclass
class DegradationResult:
    degraded: bool
    reasons: list[str] = field(default_factory=list)
    baseline_run_id: str | None = None
    baseline_missing: bool = False
    completion_drop_pp: float | None = None
    pct_drop_pp: float | None = None


@dataclass
class PipelineResult:
    config: PipelineConfig
    harness_metrics: AggregateMetrics
    evidently: RegressionResult
    degradation: DegradationResult
    alert_path: Path | None
    mlflow_run_id: str | None
    ok: bool

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1


def resolve_config(
    *,
    prompt_version: str | None = None,
    degrade_pp: float | None = None,
) -> PipelineConfig:
    version = (prompt_version or settings.PROMPT_VERSION).strip()
    path = research_prompt_path(version)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return PipelineConfig(
        prompt_version=version,
        degrade_pp=float(
            settings.REGRESSION_DEGRADE_PP if degrade_pp is None else degrade_pp
        ),
        evidently_threshold=float(settings.EVIDENTLY_PASS_THRESHOLD),
        tracking_uri=settings.MLFLOW_TRACKING_URI,
        experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
        webhook_url=(settings.REGRESSION_WEBHOOK_URL or "").strip(),
        prompt_file=path,
    )


def _rate_drop_pp(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return (float(baseline) - float(current)) * 100.0


def check_degradation(
    *,
    completion_rate: float,
    pct_tests_passed: float,
    baseline: dict[str, Any] | None,
    degrade_pp: float,
) -> DegradationResult:
    """Fail when either rate drops by more than ``degrade_pp`` percentage points.

    If there is no promoted baseline (fresh clone / empty mlruns), skip the
    compare and return ``degraded=False`` with ``baseline_missing=True``.
    """
    if baseline is None:
        return DegradationResult(
            degraded=False,
            reasons=["No promoted MLflow baseline; skip degradation compare."],
            baseline_missing=True,
        )

    reasons: list[str] = []
    completion_drop = _rate_drop_pp(completion_rate, baseline.get("completion_rate"))
    pct_drop = _rate_drop_pp(pct_tests_passed, baseline.get("pct_tests_passed"))

    if completion_drop is not None and completion_drop > degrade_pp:
        reasons.append(
            f"completion_rate dropped {completion_drop:.1f}pp "
            f"(threshold {degrade_pp:g}pp) vs run {baseline.get('run_id')}"
        )
    if pct_drop is not None and pct_drop > degrade_pp:
        reasons.append(
            f"pct_tests_passed dropped {pct_drop:.1f}pp "
            f"(threshold {degrade_pp:g}pp) vs run {baseline.get('run_id')}"
        )
    # Baseline exists but lacks comparable metrics — warn, do not fail closed.
    if completion_drop is None and pct_drop is None:
        reasons.append(
            "Promoted baseline missing completion_rate / pct_tests_passed; "
            "skip degradation compare."
        )
        return DegradationResult(
            degraded=False,
            reasons=reasons,
            baseline_run_id=str(baseline.get("run_id") or "") or None,
            baseline_missing=True,
            completion_drop_pp=completion_drop,
            pct_drop_pp=pct_drop,
        )

    return DegradationResult(
        degraded=bool(reasons),
        reasons=reasons
        or [
            f"Within {degrade_pp:g}pp of promoted baseline "
            f"(run {baseline.get('run_id')})."
        ],
        baseline_run_id=str(baseline.get("run_id") or "") or None,
        baseline_missing=False,
        completion_drop_pp=completion_drop,
        pct_drop_pp=pct_drop,
    )


def write_alert(
    degradation: DegradationResult,
    *,
    config: PipelineConfig,
    completion_rate: float,
    pct_tests_passed: float,
    path: Path | None = None,
) -> Path:
    path = path or DEFAULT_ALERT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Regression degradation alert",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Prompt version: `{config.prompt_version}`",
        f"Schedule (DAG): `{config.schedule}`",
        "",
        "## Current metrics",
        "",
        f"- completion_rate: **{completion_rate:.3f}**",
        f"- pct_tests_passed: **{pct_tests_passed:.3f}**",
        f"- degrade threshold: **{config.degrade_pp:g}pp**",
        "",
        "## Reasons",
        "",
    ]
    for reason in degradation.reasons:
        lines.append(f"- {reason}")
    if degradation.baseline_run_id:
        lines.extend(["", f"Baseline run id: `{degradation.baseline_run_id}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def post_webhook_stub(url: str, payload: dict[str, Any]) -> bool:
    """Optional HTTP POST for alerts; returns False on missing URL or network error."""
    if not url:
        return False
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=5) as resp:  # noqa: S310 — URL from env
            return 200 <= getattr(resp, "status", 200) < 300
    except (urlerror.URLError, TimeoutError, OSError) as exc:
        logger.warning("Webhook stub failed: %s", exc)
        return False


def run_scheduled_regression(
    *,
    prompt_version: str | None = None,
    degrade_pp: float | None = None,
    judge_mode: str = "scripted",
    log_mlflow: bool = True,
    write_harness_report: bool = True,
    alert_path: Path | None = None,
    simulate_degrade: bool = False,
    baseline_override: dict[str, Any] | None = None,
    harness_metrics_override: AggregateMetrics | None = None,
    evidently_override: RegressionResult | None = None,
) -> PipelineResult:
    """Full Phase 13 pipeline used by Airflow tasks and ``make airflow-dry-run``."""
    config = resolve_config(prompt_version=prompt_version, degrade_pp=degrade_pp)

    if harness_metrics_override is not None:
        harness_metrics = harness_metrics_override
        report_path = DEFAULT_REPORT if write_harness_report else None
    else:
        _, harness_metrics, _ = run_harness(write_report=write_harness_report)
        report_path = DEFAULT_REPORT if write_harness_report else None

    if evidently_override is not None:
        evidently = evidently_override
    else:
        evidently = run_regression(
            prompt_version=config.prompt_version,
            judge_mode=judge_mode,  # type: ignore[arg-type]
            log_mlflow=False,
        )

    completion_rate = float(harness_metrics.completion_rate)
    pct_tests_passed = float(evidently.pct_tests_passed)
    if simulate_degrade:
        completion_rate = max(0.0, completion_rate - (config.degrade_pp + 1) / 100.0)
        pct_tests_passed = max(0.0, pct_tests_passed - (config.degrade_pp + 1) / 100.0)

    if baseline_override is not None:
        baseline = baseline_override
    else:
        baseline = get_last_promoted_metrics(
            experiment_name=config.experiment_name,
            tracking_uri=config.tracking_uri,
        )

    degradation = check_degradation(
        completion_rate=completion_rate,
        pct_tests_passed=pct_tests_passed,
        baseline=baseline,
        degrade_pp=config.degrade_pp,
    )

    written_alert: Path | None = None
    if degradation.degraded:
        written_alert = write_alert(
            degradation,
            config=config,
            completion_rate=completion_rate,
            pct_tests_passed=pct_tests_passed,
            path=alert_path,
        )
        post_webhook_stub(
            config.webhook_url,
            {
                "event": "regression_degradation",
                "prompt_version": config.prompt_version,
                "completion_rate": completion_rate,
                "pct_tests_passed": pct_tests_passed,
                "reasons": degradation.reasons,
                "alert_path": str(written_alert),
            },
        )

    mlflow_run_id: str | None = None
    if log_mlflow:
        configure_mlflow(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
        )
        with start_experiment_run(
            config.prompt_version,
            params={
                "REGRESSION_DEGRADE_PP": config.degrade_pp,
                "EVIDENTLY_PASS_THRESHOLD": config.evidently_threshold,
                "schedule": config.schedule,
                "simulate_degrade": simulate_degrade,
                "judge_mode": judge_mode,
            },
            run_name=f"scheduled-regression-{config.prompt_version}",
            tags={
                "phase": "13",
                "component": "scheduled_regression",
                "degraded": "true" if degradation.degraded else "false",
            },
        ) as run:
            import mlflow as _mlflow

            mlflow_run_id = run.info.run_id
            log_harness_metrics(harness_metrics)
            # When simulating, overwrite rates so the logged run shows the drop.
            if simulate_degrade:
                _mlflow.log_metric("completion_rate", completion_rate)
            log_evidently_metrics(
                pct_tests_passed if simulate_degrade else evidently.pct_tests_passed,
                per_check=evidently.per_check,
                promoted=evidently.promoted and not degradation.degraded,
            )
            log_evidently_report(evidently.html_path, notes_path=evidently.notes_path)
            report_for_log = (
                report_path if report_path and Path(report_path).is_file() else None
            )
            log_prompt_and_traces(config.prompt_file, [], report_path=report_for_log)
            if written_alert and written_alert.is_file():
                log_evidently_report(written_alert, artifact_subdir="artifacts/alerts")

    ok = (
        evidently.promoted
        and not degradation.degraded
        and evidently.pct_tests_passed >= config.evidently_threshold
    )
    # When simulating, promotion from Evidently may still be true on raw suite —
    # degrade branch must fail the pipeline.
    if simulate_degrade and degradation.degraded:
        ok = False

    return PipelineResult(
        config=config,
        harness_metrics=harness_metrics,
        evidently=evidently,
        degradation=degradation,
        alert_path=written_alert,
        mlflow_run_id=mlflow_run_id,
        ok=ok,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 13 regression pipeline (Airflow dry-run / shared callables)"
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Prompt version (default: settings.PROMPT_VERSION)",
    )
    parser.add_argument(
        "--degrade-pp",
        type=float,
        default=None,
        help="Degradation threshold in percentage points (default: REGRESSION_DEGRADE_PP)",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("scripted", "llm"),
        default="scripted",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--simulate-degrade",
        action="store_true",
        help="Artificially worsen rates to exercise the degradation branch",
    )
    parser.add_argument(
        "--alert-path",
        type=Path,
        default=None,
        help="Where to write regression_alert.md on degrade",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    result = run_scheduled_regression(
        prompt_version=args.prompt_version or "prompt_v3",
        degrade_pp=args.degrade_pp,
        judge_mode=args.judge_mode,
        log_mlflow=not args.no_mlflow,
        simulate_degrade=args.simulate_degrade,
        alert_path=args.alert_path,
    )
    deg = result.degradation
    completion_shown = float(result.harness_metrics.completion_rate)
    pct_shown = float(result.evidently.pct_tests_passed)
    if args.simulate_degrade and deg.degraded:
        # Rates used for the compare were artificially worsened.
        drop = (result.config.degrade_pp + 1) / 100.0
        completion_shown = max(0.0, completion_shown - drop)
        pct_shown = max(0.0, pct_shown - drop)
    print(
        f"prompt={result.config.prompt_version} "
        f"completion_rate={completion_shown:.3f} "
        f"pct_tests_passed={pct_shown:.3f} "
        f"promoted={result.evidently.promoted} "
        f"degraded={deg.degraded} ok={result.ok}"
    )
    for reason in deg.reasons:
        print(f"  - {reason}")
    if result.alert_path:
        print(f"alert={result.alert_path}")
    if result.mlflow_run_id:
        print(f"mlflow_run_id={result.mlflow_run_id}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
