"""Airflow DAG: nightly verified-research regression eval (Phase 13).

Schedule: cron ``0 2 * * *`` (02:00 UTC daily). Catchup disabled.

Graders do **not** need a full Airflow cluster — run the same callables via::

    make airflow-dry-run

To use with a local Airflow install, point ``AIRFLOW__CORE__DAGS_FOLDER``
(or copy this file) at ``mlops/airflow/dags/``. Apache Airflow is an optional
host dependency and is not part of ``uv sync --extra mlops``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

# Shared pipeline — also used by ``python -m mlops.regression_pipeline``.
from mlops.regression_pipeline import (
    DEFAULT_SCHEDULE,
    check_degradation,
    resolve_config,
    run_scheduled_regression,
    write_alert,
)
from mlops.tracking import get_last_promoted_metrics

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover — dry-run / unit tests without Airflow
    DAG = None  # type: ignore[misc, assignment]
    PythonOperator = None  # type: ignore[misc, assignment]


def task_ensure_config(**_context: Any) -> dict[str, Any]:
    cfg = resolve_config(prompt_version=None)
    return {
        "prompt_version": cfg.prompt_version,
        "prompt_file": str(cfg.prompt_file),
        "degrade_pp": cfg.degrade_pp,
        "schedule": cfg.schedule,
        "tracking_uri": cfg.tracking_uri,
        "experiment_name": cfg.experiment_name,
    }


def task_run_full_pipeline(**_context: Any) -> dict[str, Any]:
    """Harness → Evidently → MLflow → degrade check (single callable for DAG)."""
    result = run_scheduled_regression(prompt_version=None, judge_mode="scripted")
    if not result.ok:
        reasons = "; ".join(result.degradation.reasons) or "pipeline failed"
        raise RuntimeError(
            f"Scheduled regression failed (ok=False): {reasons}"
        )
    return {
        "prompt_version": result.config.prompt_version,
        "completion_rate": result.harness_metrics.completion_rate,
        "pct_tests_passed": result.evidently.pct_tests_passed,
        "promoted": result.evidently.promoted,
        "degraded": result.degradation.degraded,
        "mlflow_run_id": result.mlflow_run_id,
        "ok": result.ok,
    }


# Re-export helpers so TaskFlow / docs can import from the DAG module if needed.
__all__ = [
    "dag",
    "task_ensure_config",
    "task_run_full_pipeline",
    "check_degradation",
    "write_alert",
    "get_last_promoted_metrics",
    "resolve_config",
]


dag = None
if DAG is not None and PythonOperator is not None:
    default_args = {
        "owner": "mlops",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 0,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="verified_research_regression_eval",
        description=(
            "Nightly eval harness + Evidently + MLflow; fail on metric degradation "
            "vs last promoted run"
        ),
        default_args=default_args,
        schedule=DEFAULT_SCHEDULE,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "regression", "verified-research"],
    ) as dag:
        ensure = PythonOperator(
            task_id="ensure_config",
            python_callable=task_ensure_config,
        )
        pipeline = PythonOperator(
            task_id="run_harness_evidently_mlflow_degrade",
            python_callable=task_run_full_pipeline,
        )
        ensure >> pipeline
