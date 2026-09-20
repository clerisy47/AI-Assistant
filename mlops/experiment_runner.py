"""Run prompt/config matrix through the scripted harness and log to MLflow."""

from __future__ import annotations

import argparse
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from app.agent.prompts import research_prompt_path
from app.agent.trace_schema import write_trace_jsonl
from app.config import settings
from eval.harness import DEFAULT_CASES, load_cases, render_report
from eval.scoring import AggregateMetrics, CaseExpect, CaseScore, aggregate, score_case
from eval.scripts import run_scenario_sync
from mlops.tracking import (
    configure_mlflow,
    log_harness_metrics,
    log_prompt_and_traces,
    start_experiment_run,
)

MLOPS_DIR = Path(__file__).resolve().parent
DEFAULT_MATRIX = MLOPS_DIR / "experiment_matrix.yaml"
DEFAULT_REPORT = MLOPS_DIR / "reports" / "mlflow_comparison.md"
REPO_ROOT = MLOPS_DIR.parent


@dataclass
class VersionSummary:
    prompt_version: str
    description: str
    config: dict[str, Any]
    metrics: AggregateMetrics
    run_id: str | None = None

    @property
    def completion_rate(self) -> float:
        return self.metrics.completion_rate

    @property
    def mean_tokens(self) -> float:
        return self.metrics.mean_tokens


@contextmanager
def apply_settings_overrides(overrides: dict[str, Any]) -> Iterator[None]:
    """Temporarily setattr on the global settings singleton."""
    originals: dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise KeyError(f"Unknown settings key: {key}")
        originals[key] = getattr(settings, key)
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(settings, key, value)


def load_matrix(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not data or "versions" not in data:
        raise ValueError(f"No versions found in {path}")
    return list(data["versions"])


def _logged_config_snapshot(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    snap = {
        "TOP_K": settings.TOP_K,
        "CHUNK_SIZE": settings.CHUNK_SIZE,
        "TEMPERATURE": settings.TEMPERATURE,
        "MAX_RESEARCH_ITERATIONS": settings.MAX_RESEARCH_ITERATIONS,
        "MAX_RESEARCH_TOOL_CALLS": settings.MAX_RESEARCH_TOOL_CALLS,
        "MAX_RESEARCH_PASS_ITERATIONS": settings.MAX_RESEARCH_PASS_ITERATIONS,
        "TOOL_RESULT_MAX_CHARS": settings.TOOL_RESULT_MAX_CHARS,
        "PROMPT_VERSION": settings.PROMPT_VERSION,
    }
    if extra:
        snap.update(extra)
    return snap


def run_harness_collecting_results(
    cases: list[dict[str, Any]],
) -> tuple[list[CaseScore], dict[str, dict[str, Any]]]:
    scores: list[CaseScore] = []
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case["id"])
        query = str(case["query"])
        scenario = str(case["scenario"])
        expect = CaseExpect.from_dict(case.get("expect"))
        try:
            result, tokens_multi, tokens_single = run_scenario_sync(
                scenario,
                query,
                compare_baseline=expect.compare_baseline,
            )
            results[case_id] = result
            scores.append(
                score_case(
                    case_id,
                    result,
                    expect,
                    tokens_multi=tokens_multi,
                    tokens_single=tokens_single,
                )
            )
        except Exception as exc:  # noqa: BLE001
            scores.append(score_case(case_id, None, expect, error=str(exc)))
    return scores, results


def write_case_traces(
    *,
    out_dir: Path,
    results: dict[str, dict[str, Any]],
    artifact_case_ids: list[str],
    prompt_version: str,
    config: dict[str, Any],
) -> list[Path]:
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for case_id in artifact_case_ids:
        result = results.get(case_id)
        if result is None:
            continue
        path = out_dir / f"{case_id}.jsonl"
        write_trace_jsonl(
            path,
            tool_trace=result.get("tool_trace"),
            iterations=int(result.get("iterations") or 0),
            stop_reason=str(result.get("stop_reason") or ""),
            token_usage=result.get("token_usage")
            if isinstance(result.get("token_usage"), dict)
            else None,
            prompt_version=prompt_version,
            config=config,
        )
        paths.append(path)
    return paths


def pick_winner(summaries: list[VersionSummary]) -> VersionSummary:
    """Prefer higher completion, then lower mean tokens, then later matrix row."""
    return max(
        enumerate(summaries),
        key=lambda pair: (pair[1].completion_rate, -pair[1].mean_tokens, pair[0]),
    )[1]


def render_comparison(summaries: list[VersionSummary], winner: VersionSummary) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# MLflow experiment comparison (Phase 11)",
        "",
        f"Generated: {now}",
        "",
        "Scripted harness runs over [`eval/cases.yaml`](../../eval/cases.yaml) "
        "with versioned prompts under [`prompts/`](../../prompts/) and config "
        "overrides from [`experiment_matrix.yaml`](../experiment_matrix.yaml).",
        "",
        "## Results",
        "",
        "| prompt | config overrides | completion | tool ok | mean iter | mean tokens | soft fails | run_id |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        cfg = ", ".join(f"{k}={v}" for k, v in s.config.items()) or "(defaults)"
        soft = s.metrics.failure_counts.get("soft", 0)
        lines.append(
            f"| `{s.prompt_version}` | {cfg} | {s.metrics.completion_rate:.0%} | "
            f"{s.metrics.tool_correctness_rate:.0%} | {s.metrics.mean_iterations:.2f} | "
            f"{s.metrics.mean_tokens:.1f} | {soft} | `{s.run_id or 'n/a'}` |"
        )

    tradeoff = (
        f"**Winner:** `{winner.prompt_version}` "
        f"(completion={winner.completion_rate:.0%}, mean_tokens={winner.mean_tokens:.1f}). "
    )
    if winner.prompt_version == "prompt_v3":
        tradeoff += (
            "v3 keeps v2’s retrieval depth and draft-naming rules while tightening "
            "`TOOL_RESULT_MAX_CHARS` for leaner context; trade-off is less raw chunk "
            "text in the research agent window versus v1 defaults."
        )
    elif winner.prompt_version == "prompt_v2":
        tradeoff += (
            "v2 improves draft specificity guidance and raises `TOP_K`; trade-off is "
            "potentially higher retrieval/token cost versus v1."
        )
    else:
        tradeoff += (
            "Baseline wins on this scripted suite; later prompt revisions target "
            "live-LLM soft failures documented in `prompts/CHANGELOG.md`."
        )

    lines.extend(
        [
            "",
            "## Winner and trade-off",
            "",
            tradeoff,
            "",
            "## Notes",
            "",
            "- Traces are custom JSONL artifacts (not MLflow autolog).",
            "- Prompt text changes do not alter scripted provider queues; config "
            "params still differ per run for comparable MLflow params/metrics.",
            "- Diagnoses that drove v2/v3: see [`prompts/CHANGELOG.md`](../../prompts/CHANGELOG.md).",
            "- Reload locally: `MLFLOW_TRACKING_URI=./mlruns mlflow ui` after "
            "`make mlflow-experiment`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_experiment(
    *,
    matrix_path: Path = DEFAULT_MATRIX,
    cases_path: Path = DEFAULT_CASES,
    comparison_path: Path = DEFAULT_REPORT,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> list[VersionSummary]:
    configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)
    versions = load_matrix(matrix_path)
    cases = load_cases(cases_path)
    summaries: list[VersionSummary] = []

    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mlops_traces_") as tmp:
        tmp_root = Path(tmp)
        for row in versions:
            prompt_version = str(row["prompt_version"])
            description = str(row.get("description") or "")
            config_overrides = dict(row.get("config") or {})
            artifact_cases = list(row.get("artifact_cases") or [])

            overrides = {
                "PROMPT_VERSION": prompt_version,
                **config_overrides,
            }
            params = {
                "TOP_K": config_overrides.get("TOP_K", settings.TOP_K),
                "TOOL_RESULT_MAX_CHARS": config_overrides.get(
                    "TOOL_RESULT_MAX_CHARS", settings.TOOL_RESULT_MAX_CHARS
                ),
                "CHUNK_SIZE": settings.CHUNK_SIZE,
                "MAX_RESEARCH_ITERATIONS": settings.MAX_RESEARCH_ITERATIONS,
                "MAX_RESEARCH_TOOL_CALLS": settings.MAX_RESEARCH_TOOL_CALLS,
                "description": description,
            }

            with apply_settings_overrides(overrides):
                # Re-resolve params that may have been overridden
                params["TOP_K"] = settings.TOP_K
                params["TOOL_RESULT_MAX_CHARS"] = settings.TOOL_RESULT_MAX_CHARS
                scores, results = run_harness_collecting_results(cases)
                metrics = aggregate(scores)
                harness_md = render_report(scores, metrics)
                harness_report = tmp_root / f"{prompt_version}_harness.md"
                harness_report.write_text(harness_md, encoding="utf-8")

                config_snap = _logged_config_snapshot(config_overrides)
                trace_dir = tmp_root / prompt_version
                trace_paths = write_case_traces(
                    out_dir=trace_dir,
                    results=results,
                    artifact_case_ids=artifact_cases,
                    prompt_version=prompt_version,
                    config=config_snap,
                )

                prompt_file = research_prompt_path(prompt_version)
                with start_experiment_run(
                    prompt_version,
                    params={
                        k: v
                        for k, v in params.items()
                        if k != "description"
                    },
                    run_name=prompt_version,
                    tags={"description": description[:250]},
                ) as run:
                    log_harness_metrics(metrics)
                    log_prompt_and_traces(prompt_file, trace_paths, harness_report)
                    run_id = run.info.run_id

                # Persist a copy of representative traces under mlops/reports for the repo
                reports_traces = comparison_path.parent / "traces" / prompt_version
                reports_traces.mkdir(parents=True, exist_ok=True)
                for tp in trace_paths:
                    dest = reports_traces / tp.name
                    dest.write_text(tp.read_text(encoding="utf-8"), encoding="utf-8")

            summaries.append(
                VersionSummary(
                    prompt_version=prompt_version,
                    description=description,
                    config=config_overrides,
                    metrics=metrics,
                    run_id=run_id,
                )
            )

    winner = pick_winner(summaries)
    markdown = render_comparison(summaries, winner)
    comparison_path.write_text(markdown, encoding="utf-8")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 11: run prompt/config matrix → MLflow + comparison report"
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override MLFLOW_TRACKING_URI (default from settings / ./mlruns)",
    )
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args(argv)

    summaries = run_experiment(
        matrix_path=args.matrix,
        cases_path=args.cases,
        comparison_path=args.report,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )
    winner = pick_winner(summaries)
    print(f"Wrote {args.report}")
    print(
        f"Winner: {winner.prompt_version} "
        f"(completion={winner.completion_rate:.0%}, mean_tokens={winner.mean_tokens:.1f})"
    )
    for s in summaries:
        print(
            f"  {s.prompt_version}: completion={s.metrics.completion_rate:.0%} "
            f"tokens={s.metrics.mean_tokens:.1f} run_id={s.run_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
