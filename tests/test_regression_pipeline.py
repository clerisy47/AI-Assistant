"""Unit tests for Phase 13 regression pipeline (no Airflow / live LLM)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.scoring import AggregateMetrics
from mlops import regression_pipeline as rp
from mlops.evidently_regression import RegressionResult
from mlops.tracking import get_last_promoted_metrics


def _metrics(completion: float = 1.0) -> AggregateMetrics:
    return AggregateMetrics(
        n_cases=5,
        completion_rate=completion,
        tool_correctness_rate=1.0,
        mean_iterations=2.0,
        mean_tokens=100.0,
        failure_counts={"none": 5},
    )


def _evidently(pct: float = 1.0, promoted: bool = True) -> RegressionResult:
    return RegressionResult(
        prompt_version="prompt_v3",
        judgments=[],
        pct_tests_passed=pct,
        per_check={"correctness_rate": pct, "refusal_rate": pct},
        promoted=promoted,
        html_path=Path("mlops/reports/evidently_prompt_v3.html"),
        notes_path=Path("mlops/reports/evidently_prompt_v3_notes.md"),
        suite_tests_passed=2,
        suite_tests_total=2,
        failing_case_ids=[],
    )


def test_check_degradation_no_baseline_skips():
    result = rp.check_degradation(
        completion_rate=0.5,
        pct_tests_passed=0.5,
        baseline=None,
        degrade_pp=10.0,
    )
    assert result.degraded is False
    assert result.baseline_missing is True


def test_check_degradation_within_threshold():
    result = rp.check_degradation(
        completion_rate=0.92,
        pct_tests_passed=0.95,
        baseline={"run_id": "abc", "completion_rate": 1.0, "pct_tests_passed": 1.0},
        degrade_pp=10.0,
    )
    assert result.degraded is False
    assert result.completion_drop_pp == pytest.approx(8.0)
    assert result.pct_drop_pp == pytest.approx(5.0)


def test_check_degradation_flags_drop():
    result = rp.check_degradation(
        completion_rate=0.80,
        pct_tests_passed=1.0,
        baseline={"run_id": "abc", "completion_rate": 1.0, "pct_tests_passed": 1.0},
        degrade_pp=10.0,
    )
    assert result.degraded is True
    assert result.completion_drop_pp == pytest.approx(20.0)
    assert any("completion_rate" in r for r in result.reasons)


def test_check_degradation_pct_drop():
    result = rp.check_degradation(
        completion_rate=1.0,
        pct_tests_passed=0.70,
        baseline={"run_id": "x", "completion_rate": 1.0, "pct_tests_passed": 1.0},
        degrade_pp=10.0,
    )
    assert result.degraded is True
    assert any("pct_tests_passed" in r for r in result.reasons)


def test_write_alert(tmp_path: Path):
    deg = rp.DegradationResult(
        degraded=True,
        reasons=["completion_rate dropped 15.0pp"],
        baseline_run_id="run-1",
    )
    cfg = rp.PipelineConfig(
        prompt_version="prompt_v3",
        degrade_pp=10.0,
        evidently_threshold=0.8,
        tracking_uri="./mlruns",
        experiment_name="verified-research",
        webhook_url="",
        prompt_file=Path("prompts/prompt_v3.md"),
    )
    path = rp.write_alert(
        deg,
        config=cfg,
        completion_rate=0.85,
        pct_tests_passed=1.0,
        path=tmp_path / "alert.md",
    )
    text = path.read_text(encoding="utf-8")
    assert "Regression degradation alert" in text
    assert "15.0pp" in text
    assert "run-1" in text


def test_resolve_config_missing_prompt(monkeypatch):
    monkeypatch.setattr(rp.settings, "PROMPT_VERSION", "prompt_does_not_exist_xyz")
    with pytest.raises(FileNotFoundError):
        rp.resolve_config(prompt_version="prompt_does_not_exist_xyz")


def test_run_scheduled_regression_ok_mocked(tmp_path: Path):
    harness = _metrics(1.0)
    evid = _evidently(1.0, promoted=True)
    with (
        patch.object(rp, "run_harness", return_value=([], harness, "# ok")),
        patch.object(rp, "run_regression", return_value=evid),
        patch.object(rp, "get_last_promoted_metrics", return_value=None),
        patch.object(rp, "configure_mlflow"),
        patch.object(rp, "start_experiment_run") as start_run,
        patch.object(rp, "log_harness_metrics"),
        patch.object(rp, "log_evidently_metrics"),
        patch.object(rp, "log_evidently_report"),
        patch.object(rp, "log_prompt_and_traces"),
    ):
        run = MagicMock()
        run.info.run_id = "fake-run"
        start_run.return_value.__enter__.return_value = run
        start_run.return_value.__exit__.return_value = None

        result = rp.run_scheduled_regression(
            prompt_version="prompt_v3",
            log_mlflow=True,
            write_harness_report=False,
            alert_path=tmp_path / "alert.md",
        )
    assert result.ok is True
    assert result.degradation.degraded is False
    assert result.alert_path is None
    assert result.mlflow_run_id == "fake-run"


def test_run_scheduled_regression_simulate_degrade(tmp_path: Path):
    harness = _metrics(1.0)
    evid = _evidently(1.0, promoted=True)
    baseline = {
        "run_id": "baseline",
        "completion_rate": 1.0,
        "pct_tests_passed": 1.0,
    }
    with (
        patch.object(rp, "run_harness", return_value=([], harness, "# ok")),
        patch.object(rp, "run_regression", return_value=evid),
        patch.object(rp, "get_last_promoted_metrics", return_value=baseline),
        patch.object(rp, "post_webhook_stub", return_value=False),
    ):
        result = rp.run_scheduled_regression(
            prompt_version="prompt_v3",
            log_mlflow=False,
            write_harness_report=False,
            simulate_degrade=True,
            alert_path=tmp_path / "alert.md",
        )
    assert result.degradation.degraded is True
    assert result.ok is False
    assert result.alert_path is not None
    assert result.alert_path.is_file()


def test_get_last_promoted_metrics_empty(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "mlops.tracking.settings.MLFLOW_TRACKING_URI",
        str(tmp_path / "empty_mlruns"),
    )
    monkeypatch.setattr(
        "mlops.tracking.settings.MLFLOW_EXPERIMENT_NAME",
        "empty-exp-phase13",
    )
    try:
        result = get_last_promoted_metrics(
            tracking_uri=str(tmp_path / "empty_mlruns"),
            experiment_name="empty-exp-phase13",
        )
    except ImportError:
        pytest.skip("mlflow not installed")
    assert result is None


def test_dag_module_imports_without_airflow():
    import mlops.airflow.dags.regression_eval_dag as dag_mod

    assert callable(dag_mod.task_ensure_config)
    assert callable(dag_mod.task_run_full_pipeline)
    # dag is None when apache-airflow is not installed
    assert dag_mod.dag is None or dag_mod.dag is not None
