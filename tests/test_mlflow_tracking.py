"""Tests for MLflow tracking helpers and trace normalization (Phase 11)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.agent.trace_schema import (
    build_run_footer,
    normalize_step,
    normalize_trace,
    write_trace_jsonl,
)
from eval.scoring import AggregateMetrics
from mlops import tracking


def test_normalize_step_maps_arguments_to_args():
    entry = {
        "agent": "research",
        "tool": "search_knowledge_base",
        "arguments": {"query": "vLLM", "top_k": 4},
        "result": "hit",
        "is_error": False,
    }
    step = normalize_step(entry, 1)
    assert step["step"] == 1
    assert step["args"] == {"query": "vLLM", "top_k": 4}
    assert step["decision"] == "search"
    assert "arguments" not in step


def test_normalize_trace_assigns_monotonic_steps():
    steps = normalize_trace(
        [
            {"agent": "research", "tool": "update_evidence_notes", "arguments": {}},
            {"agent": "verifier", "tool": "verify_claims", "arguments": {}},
        ]
    )
    assert [s["step"] for s in steps] == [1, 2]
    assert steps[1]["decision"] == "verify"


def test_write_trace_jsonl(tmp_path: Path):
    path = tmp_path / "case.jsonl"
    write_trace_jsonl(
        path,
        tool_trace=[
            {
                "agent": "research",
                "tool": "search_knowledge_base",
                "arguments": {"query": "x"},
                "result": "y",
                "is_error": False,
            }
        ],
        iterations=2,
        stop_reason="verified",
        token_usage={"total_tokens": 100},
        prompt_version="prompt_v1",
        config={"TOP_K": 4},
    )
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    step = json.loads(lines[0])
    footer = json.loads(lines[1])
    assert step["tool"] == "search_knowledge_base"
    assert footer["type"] == "run_footer"
    assert footer["stop_reason"] == "verified"
    assert footer["prompt_version"] == "prompt_v1"


def test_build_run_footer():
    footer = build_run_footer(
        iterations=3,
        stop_reason="max_iterations",
        token_usage={"total_tokens": 50},
        prompt_version="prompt_v2",
    )
    assert footer["iterations"] == 3
    assert footer["type"] == "run_footer"


def test_configure_and_log_metrics_with_mocked_mlflow():
    metrics = AggregateMetrics(
        n_cases=2,
        completion_rate=0.5,
        tool_correctness_rate=1.0,
        mean_iterations=2.0,
        mean_tokens=100.0,
        failure_counts={"none": 1, "soft": 1, "hard": 0, "cascading_soft": 0},
    )
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "abc123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with patch.object(tracking, "_import_mlflow", return_value=mock_mlflow):
        tracking.configure_mlflow(tracking_uri="./mlruns", experiment_name="test-exp")
        mock_mlflow.set_tracking_uri.assert_called_with("./mlruns")
        mock_mlflow.set_experiment.assert_called_with("test-exp")

        with tracking.start_experiment_run("prompt_v1", {"TOP_K": 4}):
            tracking.log_harness_metrics(metrics)

        mock_mlflow.log_params.assert_called()
        assert mock_mlflow.log_metric.call_count >= 4


def test_log_prompt_and_traces(tmp_path: Path):
    prompt = tmp_path / "prompt_v1.md"
    prompt.write_text("hello", encoding="utf-8")
    trace = tmp_path / "case.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "report.md"
    report.write_text("# r\n", encoding="utf-8")

    mock_mlflow = MagicMock()
    with patch.object(tracking, "_import_mlflow", return_value=mock_mlflow):
        tracking.log_prompt_and_traces(prompt, [trace], report)
        assert mock_mlflow.log_artifact.call_count == 3


def test_mlflow_missing_raises_clear_error():
    def boom():
        raise ImportError(
            "mlflow is required for experiment tracking. "
            "Install with: uv sync --extra mlops"
        )

    with patch.object(tracking, "_import_mlflow", side_effect=boom):
        with pytest.raises(ImportError, match="uv sync --extra mlops"):
            tracking.configure_mlflow()
