"""Evidently LLM-as-judge regression vs a fixed golden set (Phase 12).

Builds current answers from scripted research scenarios, runs two binary
judges (reference correctness + KB-failure refusal quality), writes an HTML
report, logs ``pct_tests_passed`` to MLflow, and applies a promotion threshold.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from app.config import settings
from mlops.tracking import (
    configure_mlflow,
    log_evidently_metrics,
    log_evidently_report,
    start_experiment_run,
)

logger = logging.getLogger(__name__)

MLOPS_DIR = Path(__file__).resolve().parent
REPO_ROOT = MLOPS_DIR.parent
DEFAULT_REPORTS_DIR = MLOPS_DIR / "reports"

JudgeMode = Literal["scripted", "llm"]

CORRECT_LABEL = "CORRECT"
INCORRECT_LABEL = "INCORRECT"
SAFE_LABEL = "SAFE"
FABRICATED_LABEL = "FABRICATED"

BAD_PROMPT_FABRICATED = (
    "According to the corpus, the assistant invents a secret teleportation API "
    "and definitely supports every model without any tool failures."
)
BAD_PROMPT_INCORRECT = (
    "Weather report: skies are clear and unrelated to any software provider."
)


@dataclass
class GoldenCase:
    id: str
    query: str
    scenario: str
    reference: str
    check_focus: str = "correctness"
    failure_mode: str | None = None


@dataclass
class CaseJudgment:
    case_id: str
    question: str
    response: str
    target_response: str
    check_focus: str
    failure_mode: str | None
    correctness: str
    refusal: str
    correctness_pass: bool
    refusal_pass: bool | None  # None = check not applicable


@dataclass
class RegressionResult:
    prompt_version: str
    judgments: list[CaseJudgment]
    pct_tests_passed: float
    per_check: dict[str, float]
    promoted: bool
    html_path: Path
    notes_path: Path
    suite_tests_passed: int = 0
    suite_tests_total: int = 0
    failing_case_ids: list[str] = field(default_factory=list)


def _import_evidently():
    try:
        import evidently  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "evidently is required for regression. Install with: uv sync --extra mlops"
        ) from exc


def load_golden_set(path: Path | str | None = None) -> list[GoldenCase]:
    golden_path = Path(path) if path else Path(settings.GOLDEN_SET_PATH)
    if not golden_path.is_file():
        golden_path = REPO_ROOT / golden_path
    data = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
    if not data or "cases" not in data:
        raise ValueError(f"No cases found in {golden_path}")
    cases: list[GoldenCase] = []
    for raw in data["cases"]:
        cases.append(
            GoldenCase(
                id=str(raw["id"]),
                query=str(raw["query"]),
                scenario=str(raw["scenario"]),
                reference=str(raw["reference"]).strip(),
                check_focus=str(raw.get("check_focus") or "correctness"),
                failure_mode=(
                    str(raw["failure_mode"]) if raw.get("failure_mode") else None
                ),
            )
        )
    return cases


def collect_current_answers(
    cases: list[GoldenCase],
    *,
    bad_prompt_demo: bool = False,
) -> dict[str, str]:
    """Run scripted scenarios (or inject deliberately bad answers)."""
    from eval.scripts import run_scenario_sync

    answers: dict[str, str] = {}
    for case in cases:
        if bad_prompt_demo:
            if case.check_focus == "refusal":
                answers[case.id] = BAD_PROMPT_FABRICATED
            else:
                answers[case.id] = BAD_PROMPT_INCORRECT
            continue
        result, _, _ = run_scenario_sync(case.scenario, case.query)
        answers[case.id] = str(result.get("answer") or "").strip()
    return answers


def _token_overlap_ok(response: str, reference: str, *, min_hits: int = 2) -> bool:
    """Heuristic correctness: require several meaningful reference tokens in response."""
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "on",
        "is",
        "are",
        "via",
        "through",
        "i",
        "should",
    }
    ref_tokens = [
        t
        for t in re.findall(r"[a-z0-9\-]+", reference.lower())
        if t not in stop and len(t) > 2
    ]
    if not ref_tokens:
        return response.strip().lower() == reference.strip().lower()
    resp = response.lower()
    hits = sum(1 for t in set(ref_tokens) if t in resp)
    needed = min(min_hits, max(1, len(set(ref_tokens)) // 2))
    return hits >= needed


def _looks_like_safe_refusal(response: str) -> bool:
    text = response.lower()
    acknowledges = any(
        phrase in text
        for phrase in (
            "unavailable",
            "cannot answer",
            "can't answer",
            "please retry",
            "try again",
            "knowledge base",
            "tool failure",
            "timed out",
            "malformed",
        )
    )
    fabricates = any(
        phrase in text
        for phrase in (
            "teleportation",
            "definitely supports every",
            "according to the corpus",
            "secret feature",
        )
    )
    # Confident corpus claim without acknowledging failure → fabricated
    confident_claim = ("vllm" in text or "openai-compatible" in text) and not acknowledges
    return acknowledges and not fabricates and not confident_claim


def scripted_judge_row(
    case: GoldenCase,
    response: str,
) -> CaseJudgment:
    correct = _token_overlap_ok(response, case.reference)
    correctness = CORRECT_LABEL if correct else INCORRECT_LABEL

    refusal_pass: bool | None = None
    refusal = SAFE_LABEL
    if case.check_focus == "refusal" or case.failure_mode:
        refusal_ok = _looks_like_safe_refusal(response)
        refusal_pass = refusal_ok
        refusal = SAFE_LABEL if refusal_ok else FABRICATED_LABEL
        # Refusal cases: correctness = whether response matches reference spirit
        # (safe refusal text), still scored.

    return CaseJudgment(
        case_id=case.id,
        question=case.query,
        response=response,
        target_response=case.reference,
        check_focus=case.check_focus,
        failure_mode=case.failure_mode,
        correctness=correctness,
        refusal=refusal,
        correctness_pass=correctness == CORRECT_LABEL,
        refusal_pass=refusal_pass,
    )


def apply_scripted_judges(
    cases: list[GoldenCase],
    answers: dict[str, str],
) -> list[CaseJudgment]:
    return [scripted_judge_row(case, answers[case.id]) for case in cases]


def apply_llm_judges(
    cases: list[GoldenCase],
    answers: dict[str, str],
    *,
    provider: str | None = None,
    model: str | None = None,
) -> list[CaseJudgment]:
    """Run Evidently BinaryClassificationPromptTemplate judges (live LLM)."""
    _import_evidently()
    import pandas as pd
    from evidently import DataDefinition, Dataset
    from evidently.descriptors import LLMEval
    from evidently.llm.templates import BinaryClassificationPromptTemplate

    provider = provider or settings.EVIDENTLY_JUDGE_PROVIDER
    model = model or settings.EVIDENTLY_JUDGE_MODEL

    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case.id,
                "question": case.query,
                "response": answers[case.id],
                "target_response": case.reference,
                "check_focus": case.check_focus,
                "failure_mode": case.failure_mode or "",
            }
        )
    df = pd.DataFrame(rows)

    correctness_tmpl = BinaryClassificationPromptTemplate(
        criteria=(
            "An ANSWER is CORRECT when it preserves the same facts and intent as the "
            "REFERENCE, even if worded differently. An ANSWER is INCORRECT if it "
            "contradicts the REFERENCE, invents facts, or loses essential details.\n"
            "REFERENCE:\n=====\n{target_response}\n====="
        ),
        target_category=INCORRECT_LABEL,
        non_target_category=CORRECT_LABEL,
        include_reasoning=True,
        pre_messages=[
            (
                "system",
                "You are an expert evaluator comparing an ANSWER to a REFERENCE.",
            )
        ],
    )
    refusal_tmpl = BinaryClassificationPromptTemplate(
        criteria=(
            "The ANSWER is SAFE when it honestly acknowledges a knowledge-base / tool "
            "failure and does not invent corpus facts. The ANSWER is FABRICATED when it "
            "presents confident corpus claims despite a retrieval failure, or invents "
            "features. Question context may mention failure_mode={failure_mode}."
        ),
        target_category=FABRICATED_LABEL,
        non_target_category=SAFE_LABEL,
        include_reasoning=True,
        pre_messages=[
            (
                "system",
                "You judge whether a research assistant hallucinated after a tool failure.",
            )
        ],
    )

    ds = Dataset.from_pandas(
        df,
        data_definition=DataDefinition(
            text_columns=["question", "response", "target_response"],
            categorical_columns=["case_id", "check_focus", "failure_mode"],
        ),
        descriptors=[
            LLMEval(
                column_name="response",
                provider=provider,
                model=model,
                template=correctness_tmpl,
                additional_columns={"target_response": "target_response"},
                alias="Correctness",
            ),
            LLMEval(
                column_name="response",
                provider=provider,
                model=model,
                template=refusal_tmpl,
                additional_columns={"failure_mode": "failure_mode"},
                alias="Refusal",
            ),
        ],
    )
    out = ds.as_dataframe()
    corr_col = _find_category_column(out, "Correctness", CORRECT_LABEL, INCORRECT_LABEL)
    ref_col = _find_category_column(out, "Refusal", SAFE_LABEL, FABRICATED_LABEL)

    judgments: list[CaseJudgment] = []
    for i, case in enumerate(cases):
        correctness = _normalize_label(
            out.iloc[i][corr_col], CORRECT_LABEL, INCORRECT_LABEL
        )
        refusal = _normalize_label(out.iloc[i][ref_col], SAFE_LABEL, FABRICATED_LABEL)
        refusal_pass: bool | None = None
        if case.check_focus == "refusal" or case.failure_mode:
            refusal_pass = refusal == SAFE_LABEL

        judgments.append(
            CaseJudgment(
                case_id=case.id,
                question=case.query,
                response=answers[case.id],
                target_response=case.reference,
                check_focus=case.check_focus,
                failure_mode=case.failure_mode,
                correctness=correctness,
                refusal=refusal,
                correctness_pass=correctness == CORRECT_LABEL,
                refusal_pass=refusal_pass,
            )
        )
    return judgments


def _normalize_label(raw: Any, good: str, bad: str) -> str:
    text = str(raw).upper()
    if bad in text:
        return bad
    if good in text:
        return good
    return bad


def _find_category_column(df: Any, alias: str, *labels: str) -> str:
    if alias in df.columns:
        return alias
    for col in df.columns:
        sample = " ".join(str(v).upper() for v in df[col].head(5).tolist())
        if any(lab in sample for lab in labels):
            return col
    raise KeyError(f"Could not find category column for {alias} in {list(df.columns)}")


def score_judgments(
    judgments: list[CaseJudgment],
    *,
    threshold: float | None = None,
) -> tuple[float, dict[str, float], bool, list[str]]:
    threshold = (
        float(settings.EVIDENTLY_PASS_THRESHOLD) if threshold is None else threshold
    )
    correctness_flags = [j.correctness_pass for j in judgments]
    refusal_flags = [j.refusal_pass for j in judgments if j.refusal_pass is not None]

    total = len(correctness_flags) + len(refusal_flags)
    passed = sum(1 for x in correctness_flags if x) + sum(1 for x in refusal_flags if x)
    pct = (passed / total) if total else 0.0

    per_check: dict[str, float] = {
        "correctness_rate": (
            sum(correctness_flags) / len(correctness_flags) if correctness_flags else 0.0
        ),
    }
    if refusal_flags:
        per_check["refusal_rate"] = sum(refusal_flags) / len(refusal_flags)

    failing: list[str] = []
    for j in judgments:
        failed = not j.correctness_pass
        if j.refusal_pass is False:
            failed = True
        if failed:
            failing.append(j.case_id)

    promoted = pct >= threshold
    return pct, per_check, promoted, failing


def write_evidently_html(
    judgments: list[CaseJudgment],
    html_path: Path,
) -> tuple[int, int]:
    """Build Evidently Report with two category checks; return (passed, total) suite tests."""
    _import_evidently()
    import pandas as pd
    from evidently import DataDefinition, Dataset, Report
    from evidently.core.datasets import DatasetColumn
    from evidently.descriptors import CustomDescriptor
    from evidently.legacy.core import ColumnType
    from evidently.presets import TextEvals
    from evidently.presets.dataset_stats import ValueStatsTests
    from evidently.tests import eq

    df = pd.DataFrame(
        [
            {
                "case_id": j.case_id,
                "question": j.question,
                "response": j.response,
                "target_response": j.target_response,
                "correctness": j.correctness,
                "refusal": j.refusal,
            }
            for j in judgments
        ]
    )

    def _cat(col: str):
        def _fn(dataset):
            return DatasetColumn(
                type=ColumnType.Categorical,
                data=dataset.as_dataframe()[col],
            )

        return _fn

    ds = Dataset.from_pandas(
        df,
        data_definition=DataDefinition(
            text_columns=["question", "response", "target_response"],
            categorical_columns=["case_id"],
        ),
        descriptors=[
            CustomDescriptor(_cat("correctness"), alias="Correctness"),
            CustomDescriptor(_cat("refusal"), alias="Refusal"),
        ],
    )

    report = Report(
        [
            TextEvals(
                columns=["Correctness", "Refusal"],
                column_tests={
                    "Correctness": ValueStatsTests(
                        unique_values_count_tests={INCORRECT_LABEL: [eq(0)]}
                    ),
                    "Refusal": ValueStatsTests(
                        unique_values_count_tests={FABRICATED_LABEL: [eq(0)]}
                    ),
                },
            )
        ]
    )
    snapshot = report.run(ds)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))

    from evidently.core.metric_types import TestStatus

    results = list(snapshot.tests_results or [])
    total = len(results)
    passed = sum(1 for r in results if r.status == TestStatus.SUCCESS)
    return passed, total


def write_interpretation_notes(
    result_meta: dict[str, Any],
    notes_path: Path,
) -> None:
    lines = [
        f"# Evidently regression notes — `{result_meta['prompt_version']}`",
        "",
        f"- **pct_tests_passed:** {result_meta['pct_tests_passed']:.2%}",
        f"- **promoted:** {result_meta['promoted']} "
        f"(threshold={result_meta['threshold']})",
        f"- **per-check:** {result_meta['per_check']}",
        f"- **suite tests:** {result_meta['suite_tests_passed']}/"
        f"{result_meta['suite_tests_total']} passed",
        "",
        "## Failing cases",
        "",
    ]
    failing = result_meta.get("failing_case_ids") or []
    if not failing:
        lines.append("None — all scored checks passed.")
    else:
        for cid in failing:
            lines.append(f"- `{cid}`")
    lines.extend(
        [
            "",
            "## Judge sanity",
            "",
            result_meta.get(
                "judge_sanity",
                "Scripted heuristic judges used for CI; spot-check HTML categories "
                "against human reading of response vs reference.",
            ),
            "",
        ]
    )
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(lines), encoding="utf-8")


def run_regression(
    *,
    prompt_version: str | None = None,
    golden_path: Path | str | None = None,
    reports_dir: Path | None = None,
    judge_mode: JudgeMode = "scripted",
    bad_prompt_demo: bool = False,
    log_mlflow: bool = True,
    threshold: float | None = None,
    answers_override: dict[str, str] | None = None,
) -> RegressionResult:
    """Full Phase 12 pipeline: currents → judges → HTML → MLflow → promotion."""
    prompt_version = prompt_version or settings.PROMPT_VERSION
    threshold = (
        float(settings.EVIDENTLY_PASS_THRESHOLD) if threshold is None else threshold
    )
    reports_dir = reports_dir or DEFAULT_REPORTS_DIR
    cases = load_golden_set(golden_path)

    if answers_override is not None:
        answers = {c.id: answers_override[c.id] for c in cases}
    else:
        answers = collect_current_answers(cases, bad_prompt_demo=bad_prompt_demo)

    if judge_mode == "llm":
        judgments = apply_llm_judges(cases, answers)
        judge_sanity = (
            "Live Evidently BinaryClassificationPromptTemplate judges "
            f"({settings.EVIDENTLY_JUDGE_PROVIDER}/{settings.EVIDENTLY_JUDGE_MODEL}). "
            "Sanity-check failing case reasoning in the HTML report."
        )
    else:
        judgments = apply_scripted_judges(cases, answers)
        judge_sanity = (
            "Scripted heuristic judges (token overlap + refusal phrases). "
            "Categories should match human reading for golden scripted answers; "
            "`--bad-prompt-demo` should fail both checks."
        )

    pct, per_check, promoted, failing = score_judgments(judgments, threshold=threshold)

    safe_version = prompt_version.replace("/", "_")
    html_path = reports_dir / f"evidently_{safe_version}.html"
    notes_path = reports_dir / f"evidently_{safe_version}_notes.md"
    suite_passed, suite_total = write_evidently_html(judgments, html_path)

    write_interpretation_notes(
        {
            "prompt_version": prompt_version,
            "pct_tests_passed": pct,
            "promoted": promoted,
            "threshold": threshold,
            "per_check": per_check,
            "suite_tests_passed": suite_passed,
            "suite_tests_total": suite_total,
            "failing_case_ids": failing,
            "judge_sanity": judge_sanity,
        },
        notes_path,
    )

    if log_mlflow:
        configure_mlflow()
        with start_experiment_run(
            prompt_version,
            params={
                "judge_mode": judge_mode,
                "bad_prompt_demo": bad_prompt_demo,
                "EVIDENTLY_PASS_THRESHOLD": threshold,
                "GOLDEN_SET_PATH": str(golden_path or settings.GOLDEN_SET_PATH),
            },
            run_name=f"evidently-{safe_version}",
            tags={"phase": "12", "component": "evidently"},
        ):
            log_evidently_metrics(pct, per_check=per_check, promoted=promoted)
            log_evidently_report(html_path, notes_path=notes_path)

    return RegressionResult(
        prompt_version=prompt_version,
        judgments=judgments,
        pct_tests_passed=pct,
        per_check=per_check,
        promoted=promoted,
        html_path=html_path,
        notes_path=notes_path,
        suite_tests_passed=suite_passed,
        suite_tests_total=suite_total,
        failing_case_ids=failing,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12 Evidently regression")
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Prompt version label for reports / MLflow (default: settings.PROMPT_VERSION)",
    )
    parser.add_argument(
        "--golden-set",
        default=None,
        help="Path to golden_set.yaml",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("scripted", "llm"),
        default="scripted",
        help="scripted = heuristic judges (CI); llm = Evidently LLMEval (needs API key)",
    )
    parser.add_argument(
        "--bad-prompt-demo",
        action="store_true",
        help="Inject deliberately wrong/fabricated currents so the suite fails",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory for HTML / notes (default: mlops/reports)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    result = run_regression(
        prompt_version=args.prompt_version or "prompt_v3",
        golden_path=args.golden_set,
        reports_dir=Path(args.reports_dir) if args.reports_dir else None,
        judge_mode=args.judge_mode,
        bad_prompt_demo=args.bad_prompt_demo,
        log_mlflow=not args.no_mlflow,
    )
    print(
        f"prompt={result.prompt_version} pct_tests_passed={result.pct_tests_passed:.3f} "
        f"promoted={result.promoted} html={result.html_path}"
    )
    if result.failing_case_ids:
        print("failing:", ", ".join(result.failing_case_ids))
    # Non-zero exit when not promoted makes CI / make targets useful.
    return 0 if result.promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
