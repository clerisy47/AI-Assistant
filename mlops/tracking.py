"""MLflow helpers for verified-research prompt/config experiments (Phase 11)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from app.config import settings
from eval.scoring import AggregateMetrics

logger = logging.getLogger(__name__)


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "mlflow is required for experiment tracking. "
            "Install with: uv sync --extra mlops"
        ) from exc
    return mlflow


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    mlflow = _import_mlflow()
    # MLflow 3.x marks the filesystem backend as maintenance-mode; allow it so
    # SPECS default MLFLOW_TRACKING_URI=./mlruns still works for local grading.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = tracking_uri if tracking_uri is not None else settings.MLFLOW_TRACKING_URI
    name = (
        experiment_name
        if experiment_name is not None
        else settings.MLFLOW_EXPERIMENT_NAME
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    logger.info("MLflow tracking_uri=%s experiment=%s", uri, name)


@contextmanager
def start_experiment_run(
    prompt_version: str,
    params: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Start an MLflow run tagged with prompt_version; yields the active run."""
    mlflow = _import_mlflow()
    merged_params = {"prompt_version": prompt_version, **(params or {})}
    # MLflow params must be strings / numbers / bools — coerce others.
    safe_params = {k: _param_value(v) for k, v in merged_params.items()}
    with mlflow.start_run(run_name=run_name or prompt_version) as run:
        mlflow.log_params(safe_params)
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("prompt_version", prompt_version)
        yield run


def _param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value)


def log_harness_metrics(metrics: AggregateMetrics) -> None:
    mlflow = _import_mlflow()
    mlflow.log_metric("completion_rate", float(metrics.completion_rate))
    mlflow.log_metric("tool_correctness_rate", float(metrics.tool_correctness_rate))
    mlflow.log_metric("mean_iterations", float(metrics.mean_iterations))
    mlflow.log_metric("mean_tokens", float(metrics.mean_tokens))
    mlflow.log_metric("n_cases", float(metrics.n_cases))
    for key, count in (metrics.failure_counts or {}).items():
        mlflow.log_metric(f"failure_{key}", float(count))


def log_prompt_and_traces(
    prompt_path: Path | str,
    trace_paths: list[Path | str],
    report_path: Path | str | None = None,
    *,
    artifact_subdir: str = "artifacts",
) -> None:
    mlflow = _import_mlflow()
    prompt = Path(prompt_path)
    if prompt.is_file():
        mlflow.log_artifact(str(prompt), artifact_path=artifact_subdir)
    for tp in trace_paths:
        p = Path(tp)
        if p.is_file():
            mlflow.log_artifact(str(p), artifact_path=f"{artifact_subdir}/traces")
    if report_path is not None:
        rp = Path(report_path)
        if rp.is_file():
            mlflow.log_artifact(str(rp), artifact_path=artifact_subdir)


def active_run_id() -> Optional[str]:
    mlflow = _import_mlflow()
    run = mlflow.active_run()
    return run.info.run_id if run is not None else None


def log_evidently_metrics(
    pct_tests_passed: float,
    *,
    per_check: dict[str, float] | None = None,
    promoted: bool,
) -> None:
    """Log Evidently pass rate + promotion flag on the active MLflow run."""
    mlflow = _import_mlflow()
    mlflow.log_metric("pct_tests_passed", float(pct_tests_passed))
    mlflow.set_tag("promoted", "true" if promoted else "false")
    if per_check:
        for name, rate in per_check.items():
            mlflow.log_metric(f"evidently_{name}", float(rate))


def log_evidently_report(
    html_path: Path | str,
    *,
    notes_path: Path | str | None = None,
    artifact_subdir: str = "artifacts",
) -> None:
    mlflow = _import_mlflow()
    html = Path(html_path)
    if html.is_file():
        mlflow.log_artifact(str(html), artifact_path=artifact_subdir)
    if notes_path is not None:
        notes = Path(notes_path)
        if notes.is_file():
            mlflow.log_artifact(str(notes), artifact_path=artifact_subdir)
