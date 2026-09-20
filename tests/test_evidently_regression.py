"""Tests for Evidently golden-set regression (Phase 12) — no live judge API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mlops import evidently_regression as er
from mlops import tracking


REQUIRED_GOLDEN_IDS = {
    "sufficient_first_search",
    "needs_re_search",
    "verify_fail_then_revise",
    "clarification",
    "kb_unavailable_recognized",
}


def test_load_golden_set_schema():
    cases = er.load_golden_set()
    ids = {c.id for c in cases}
    assert REQUIRED_GOLDEN_IDS <= ids
    for case in cases:
        assert case.query
        assert case.reference
        assert case.scenario
        assert case.check_focus in {"correctness", "refusal"}


def test_scripted_judges_promote_good_answers(tmp_path: Path):
    cases = er.load_golden_set()
    # Use references themselves as currents → should pass heuristics.
    answers = {c.id: c.reference for c in cases}
    judgments = er.apply_scripted_judges(cases, answers)
    pct, per_check, promoted, failing = er.score_judgments(judgments, threshold=0.8)
    assert pct >= 0.8
    assert promoted is True
    assert failing == []
    assert per_check["correctness_rate"] >= 0.8
    assert per_check.get("refusal_rate", 1.0) >= 0.8

    html = tmp_path / "evidently_prompt_v3.html"
    suite_passed, suite_total = er.write_evidently_html(judgments, html)
    assert html.is_file()
    assert html.stat().st_size > 100
    assert suite_total >= 1
    assert suite_passed == suite_total


def test_bad_currents_block_promotion(tmp_path: Path):
    cases = er.load_golden_set()
    answers = {
        c.id: (
            er.BAD_PROMPT_FABRICATED
            if c.check_focus == "refusal"
            else er.BAD_PROMPT_INCORRECT
        )
        for c in cases
    }
    judgments = er.apply_scripted_judges(cases, answers)
    pct, _, promoted, failing = er.score_judgments(judgments, threshold=0.8)
    assert pct < 0.8
    assert promoted is False
    assert "kb_unavailable_recognized" in failing
    assert "sufficient_first_search" in failing

    html = tmp_path / "evidently_bad.html"
    suite_passed, suite_total = er.write_evidently_html(judgments, html)
    assert html.is_file()
    assert suite_passed < suite_total


def test_run_regression_with_overrides_and_mlflow(tmp_path: Path):
    cases = er.load_golden_set()
    good = {c.id: c.reference for c in cases}

    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "ev-run-1"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = False

    with patch.object(tracking, "_import_mlflow", return_value=mock_mlflow):
        result = er.run_regression(
            prompt_version="prompt_v3",
            reports_dir=tmp_path,
            judge_mode="scripted",
            answers_override=good,
            log_mlflow=True,
        )

    assert result.promoted is True
    assert result.pct_tests_passed >= 0.8
    assert result.html_path.is_file()
    assert result.notes_path.is_file()
    notes = result.notes_path.read_text(encoding="utf-8")
    assert "pct_tests_passed" in notes
    assert "promoted" in notes

    metric_names = [c.args[0] for c in mock_mlflow.log_metric.call_args_list]
    assert "pct_tests_passed" in metric_names
    mock_mlflow.set_tag.assert_any_call("promoted", "true")
    assert mock_mlflow.log_artifact.call_count >= 1


def test_run_regression_bad_prompt_demo(tmp_path: Path):
    with patch.object(tracking, "_import_mlflow", return_value=MagicMock()):
        result = er.run_regression(
            prompt_version="prompt_bad",
            reports_dir=tmp_path,
            judge_mode="scripted",
            bad_prompt_demo=True,
            log_mlflow=False,
        )
    assert result.promoted is False
    assert result.pct_tests_passed < 0.8
    assert result.html_path.is_file()


def test_log_evidently_metrics_helper():
    mock_mlflow = MagicMock()
    with patch.object(tracking, "_import_mlflow", return_value=mock_mlflow):
        tracking.log_evidently_metrics(
            0.9,
            per_check={"correctness_rate": 1.0, "refusal_rate": 0.8},
            promoted=True,
        )
    mock_mlflow.log_metric.assert_any_call("pct_tests_passed", 0.9)
    mock_mlflow.set_tag.assert_called_with("promoted", "true")
