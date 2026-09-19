"""Eval harness runner: load cases → scripted supervisor → score → report.md."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eval.scoring import AggregateMetrics, CaseExpect, CaseScore, aggregate, score_case
from eval.scripts import run_scenario_sync

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "cases.yaml"
DEFAULT_REPORT = EVAL_DIR / "report.md"


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not data or "cases" not in data:
        raise ValueError(f"No cases found in {path}")
    return list(data["cases"])


def run_case(case: dict[str, Any]) -> CaseScore:
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
        return score_case(
            case_id,
            result,
            expect,
            tokens_multi=tokens_multi,
            tokens_single=tokens_single,
        )
    except Exception as exc:  # noqa: BLE001 — harness must classify crashes
        return score_case(case_id, None, expect, error=str(exc))


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def _fmt_tokens(score: CaseScore) -> str:
    if score.tokens_multi is not None and score.tokens_single is not None:
        return f"{score.tokens} (multi={score.tokens_multi}, single={score.tokens_single})"
    return str(score.tokens)


def render_report(scores: list[CaseScore], metrics: AggregateMetrics) -> str:
    lines: list[str] = [
        "# Verified research — eval report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Summary",
        "",
        f"- Cases: **{metrics.n_cases}**",
        f"- Task completion rate: **{metrics.completion_rate:.0%}**",
        f"- Tool-call correctness: **{metrics.tool_correctness_rate:.0%}**",
        f"- Mean iterations: **{metrics.mean_iterations:.2f}**",
        f"- Mean tokens: **{metrics.mean_tokens:.1f}**",
        f"- Failures: hard={metrics.failure_counts.get('hard', 0)}, "
        f"soft={metrics.failure_counts.get('soft', 0)}, "
        f"cascading_soft={metrics.failure_counts.get('cascading_soft', 0)}, "
        f"none={metrics.failure_counts.get('none', 0)}",
        "",
        "## Per-case results",
        "",
        "| case id | completed | tools ok | iterations | tokens | failure class | notes |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for s in scores:
        notes = (s.notes or "").replace("|", "/")
        lines.append(
            f"| {s.case_id} | {_fmt_bool(s.completed)} | {_fmt_bool(s.tools_ok)} | "
            f"{s.iterations} | {_fmt_tokens(s)} | {s.failure_class} | {notes} |"
        )

    failures = [s for s in scores if s.failure_class != "none"]
    lines.extend(["", "## Failure log", ""])
    if not failures:
        lines.append("No failures.")
    else:
        for s in failures:
            lines.append(
                f"- **{s.case_id}** (`{s.failure_class}`): stop={s.stop_reason or 'n/a'}; "
                f"{s.notes or s.error or 'see table'}"
            )

    baseline = next((s for s in scores if s.tokens_multi is not None), None)
    lines.extend(["", "## Baseline token comparison", ""])
    if baseline is None:
        lines.append("No `compare_baseline` case in this run.")
    else:
        lines.append(
            f"- Case `{baseline.case_id}`: multi-agent tokens={baseline.tokens_multi}, "
            f"single-agent tokens={baseline.tokens_single}."
        )
        if baseline.tokens_multi and baseline.tokens_single:
            delta = baseline.tokens_multi - baseline.tokens_single
            lines.append(
                f"- Coordination overhead (multi − single): **{delta}** tokens "
                "(positive means multi-agent used more)."
            )

    lines.append("")
    return "\n".join(lines)


def run_harness(
    cases_path: Path = DEFAULT_CASES,
    report_path: Path = DEFAULT_REPORT,
    *,
    write_report: bool = True,
) -> tuple[list[CaseScore], AggregateMetrics, str]:
    cases = load_cases(cases_path)
    scores = [run_case(c) for c in cases]
    metrics = aggregate(scores)
    markdown = render_report(scores, metrics)
    if write_report:
        report_path.write_text(markdown, encoding="utf-8")
    return scores, metrics, markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verified research evaluation harness")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="Path to cases.yaml",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to write Markdown report",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print report only; do not write report.md",
    )
    args = parser.parse_args(argv)

    _, _, markdown = run_harness(
        cases_path=args.cases,
        report_path=args.report,
        write_report=not args.no_write,
    )
    print(markdown)
    if not args.no_write:
        print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
