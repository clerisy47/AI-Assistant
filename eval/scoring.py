"""Pure metrics and failure taxonomy for the research eval harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

FailureClass = Literal["none", "hard", "soft", "cascading_soft"]

_EMPTY_RETRIEVAL_MARKERS = (
    "no relevant documents",
    "knowledge base unavailable",
    "timed out",
    "malformed",
)


@dataclass
class CaseExpect:
    stop_reason: str
    required_tools: list[str] = field(default_factory=list)
    answer_keywords: list[str] = field(default_factory=list)
    expected_max_iterations: int | None = None
    tool_order: list[str] | None = None
    verification_sufficient: bool | None = None
    baseline: Literal["multi", "single"] = "multi"
    compare_baseline: bool = False
    expected_failure_class: FailureClass | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> CaseExpect:
        data = dict(raw or {})
        return cls(
            stop_reason=str(data.get("stop_reason", "verified")),
            required_tools=list(data.get("required_tools") or []),
            answer_keywords=list(data.get("answer_keywords") or []),
            expected_max_iterations=data.get("expected_max_iterations"),
            tool_order=list(data["tool_order"]) if data.get("tool_order") else None,
            verification_sufficient=data.get("verification_sufficient"),
            baseline=data.get("baseline") or "multi",
            compare_baseline=bool(data.get("compare_baseline", False)),
            expected_failure_class=data.get("expected_failure_class")
            or data.get("failure_class"),
        )


@dataclass
class CaseScore:
    case_id: str
    completed: bool
    tools_ok: bool
    trajectory_ok: bool
    iterations: int
    tool_call_count: int
    tokens: int
    tokens_multi: int | None = None
    tokens_single: int | None = None
    failure_class: FailureClass = "none"
    notes: str = ""
    stop_reason: str = ""
    error: str | None = None


def _tool_names(tool_trace: list[dict[str, Any]]) -> list[str]:
    return [str(e.get("tool") or "") for e in tool_trace if e.get("tool")]


def _keywords_present(answer: str, keywords: list[str]) -> bool:
    lower = (answer or "").lower()
    return all(k.lower() in lower for k in keywords)


def _has_empty_retrieval(tool_trace: list[dict[str, Any]]) -> bool:
    for entry in tool_trace:
        if entry.get("tool") != "search_knowledge_base":
            continue
        result = str(entry.get("result") or "").lower()
        if any(m in result for m in _EMPTY_RETRIEVAL_MARKERS):
            return True
        if entry.get("is_error"):
            return True
    return False


def _evidence_item_count(result: dict[str, Any]) -> int:
    notes = result.get("evidence_notes") or {}
    items = notes.get("items") if isinstance(notes, dict) else None
    if items is None:
        return 0
    return len(items)


def _answer_is_empty(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if (
        len(lines) <= 1
        and "partial answer" in text.lower()
        and "evidence so far" not in text.lower()
    ):
        return True
    return False


def classify_failure(
    *,
    result: Optional[dict[str, Any]],
    expect: CaseExpect,
    completed: bool,
    tools_ok: bool,
    trajectory_ok: bool,
    error: str | None = None,
) -> tuple[FailureClass, str]:
    """Map SPECS hard / soft / cascading_soft taxonomy (auto; ignore expected override)."""
    if error:
        return "hard", f"crash: {error}"

    if result is None:
        return "hard", "missing result"

    stop = str(result.get("stop_reason") or "")
    answer = str(result.get("answer") or "")
    verification = result.get("verification")
    baseline = expect.baseline
    tool_trace = list(result.get("tool_trace") or [])
    evidence_n = _evidence_item_count(result)

    if stop == "verified" and baseline == "multi":
        if not isinstance(verification, dict) or verification.get("sufficient") is not True:
            return "hard", "verified stop without verification.sufficient=true"

    if stop in ("max_iterations", "max_tool_calls"):
        if _answer_is_empty(answer) and evidence_n == 0:
            return "hard", "budget stop with empty answer and no evidence"

    if _has_empty_retrieval(tool_trace) and stop == "verified" and evidence_n == 0:
        return "cascading_soft", "empty retrieval then verified with no evidence notes"

    if completed and tools_ok and trajectory_ok:
        return "none", ""

    notes_parts: list[str] = []
    if not completed:
        notes_parts.append("rubric miss")
    if not tools_ok:
        notes_parts.append("tools")
    if not trajectory_ok:
        notes_parts.append("trajectory")
    return "soft", "; ".join(notes_parts) or "soft failure"


def score_case(
    case_id: str,
    result: Optional[dict[str, Any]],
    expect: CaseExpect,
    *,
    error: str | None = None,
    tokens_multi: int | None = None,
    tokens_single: int | None = None,
) -> CaseScore:
    if error or result is None:
        return CaseScore(
            case_id=case_id,
            completed=False,
            tools_ok=False,
            trajectory_ok=False,
            iterations=0,
            tool_call_count=0,
            tokens=0,
            tokens_multi=tokens_multi,
            tokens_single=tokens_single,
            failure_class="hard",
            notes=f"crash: {error}" if error else "missing result",
            stop_reason="",
            error=error,
        )

    tool_trace = list(result.get("tool_trace") or [])
    names = _tool_names(tool_trace)
    research_tools = [n for n in names if n != "verify_claims"]
    tool_call_count = len(research_tools)
    iterations = int(result.get("iterations") or 0)
    usage = result.get("token_usage") or {}
    tokens = int(usage.get("total_tokens") or 0)
    stop = str(result.get("stop_reason") or "")
    answer = str(result.get("answer") or "")

    stop_ok = stop == expect.stop_reason
    keywords_ok = (
        _keywords_present(answer, expect.answer_keywords) if expect.answer_keywords else True
    )
    completed = stop_ok and keywords_ok

    if expect.verification_sufficient is not None:
        verification = result.get("verification") or {}
        if isinstance(verification, dict):
            completed = completed and (
                bool(verification.get("sufficient")) == expect.verification_sufficient
            )
        elif expect.verification_sufficient is True:
            completed = False

    tools_ok = all(t in names for t in expect.required_tools)
    if tools_ok and expect.tool_order:
        order_idx = 0
        for name in names:
            if order_idx < len(expect.tool_order) and name == expect.tool_order[order_idx]:
                order_idx += 1
        if order_idx < len(expect.tool_order):
            tools_ok = False

    trajectory_ok = True
    if expect.expected_max_iterations is not None:
        trajectory_ok = iterations <= int(expect.expected_max_iterations)

    auto_class, auto_notes = classify_failure(
        result=result,
        expect=expect,
        completed=completed,
        tools_ok=tools_ok,
        trajectory_ok=trajectory_ok,
        error=None,
    )

    if expect.expected_failure_class:
        failure_class = expect.expected_failure_class
        notes = f"taxonomy demo: {expect.expected_failure_class} (auto={auto_class})"
    else:
        failure_class = auto_class
        notes = auto_notes

    return CaseScore(
        case_id=case_id,
        completed=completed,
        tools_ok=tools_ok,
        trajectory_ok=trajectory_ok,
        iterations=iterations,
        tool_call_count=tool_call_count,
        tokens=tokens,
        tokens_multi=tokens_multi,
        tokens_single=tokens_single,
        failure_class=failure_class,
        notes=notes,
        stop_reason=stop,
        error=None,
    )


@dataclass
class AggregateMetrics:
    n_cases: int
    completion_rate: float
    tool_correctness_rate: float
    mean_iterations: float
    mean_tokens: float
    failure_counts: dict[str, int]


def aggregate(scores: list[CaseScore]) -> AggregateMetrics:
    n = len(scores) or 1
    completed = sum(1 for s in scores if s.completed)
    tools_ok = sum(1 for s in scores if s.tools_ok)
    mean_iter = sum(s.iterations for s in scores) / n
    mean_tok = sum(s.tokens for s in scores) / n
    counts: dict[str, int] = {"none": 0, "hard": 0, "soft": 0, "cascading_soft": 0}
    for s in scores:
        counts[s.failure_class] = counts.get(s.failure_class, 0) + 1
    return AggregateMetrics(
        n_cases=len(scores),
        completion_rate=completed / n if scores else 0.0,
        tool_correctness_rate=tools_ok / n if scores else 0.0,
        mean_iterations=mean_iter,
        mean_tokens=mean_tok,
        failure_counts=counts,
    )
