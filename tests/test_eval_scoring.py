"""Unit tests for eval scoring / failure taxonomy (no LLM)."""

from __future__ import annotations

from eval.scoring import CaseExpect, aggregate, classify_failure, score_case


def _result(
    *,
    stop: str = "verified",
    answer: str = "Supports OpenAI-compatible vLLM.",
    tools: list[str] | None = None,
    iterations: int = 1,
    sufficient: bool | None = True,
    evidence_items: list[dict] | None = None,
    kb_result: str | None = None,
) -> dict:
    tool_trace = []
    for name in tools or ["search_knowledge_base", "update_evidence_notes"]:
        entry = {
            "agent": "research",
            "tool": name,
            "arguments": {},
            "result": "ok",
            "is_error": False,
        }
        if name == "search_knowledge_base" and kb_result is not None:
            entry["result"] = kb_result
        tool_trace.append(entry)
    if sufficient is not None:
        tool_trace.append(
            {
                "agent": "verifier",
                "tool": "verify_claims",
                "arguments": {},
                "result": "{}",
                "is_error": False,
            }
        )
    verification = None
    if sufficient is not None:
        verification = {
            "sufficient": sufficient,
            "unsupported_claims": [],
            "suggested_next_action": "finalize" if sufficient else "re_research",
            "notes": "",
        }
    return {
        "answer": answer,
        "stop_reason": stop,
        "iterations": iterations,
        "tool_trace": tool_trace,
        "evidence_notes": {
            "question": "q",
            "items": evidence_items
            if evidence_items is not None
            else [{"source": "a.md", "excerpt": "x", "query_used": "q"}],
            "open_gaps": [],
            "sufficient_hypothesis": True,
        },
        "token_usage": {"total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20},
        "verification": verification,
    }


def test_score_case_pass():
    expect = CaseExpect(
        stop_reason="verified",
        required_tools=["search_knowledge_base"],
        answer_keywords=["vLLM"],
        expected_max_iterations=2,
        verification_sufficient=True,
    )
    score = score_case("ok", _result(), expect)
    assert score.completed is True
    assert score.tools_ok is True
    assert score.trajectory_ok is True
    assert score.failure_class == "none"
    assert score.tokens == 100


def test_keyword_miss_is_soft():
    expect = CaseExpect(
        stop_reason="verified",
        required_tools=["search_knowledge_base"],
        answer_keywords=["OpenAI-compatible"],
        expected_max_iterations=2,
    )
    score = score_case(
        "weak",
        _result(answer="Local models are supported somehow."),
        expect,
    )
    assert score.completed is False
    assert score.failure_class == "soft"


def test_tool_order_subsequence():
    expect = CaseExpect(
        stop_reason="verified",
        required_tools=["search_knowledge_base", "update_evidence_notes"],
        tool_order=["search_knowledge_base", "update_evidence_notes"],
        answer_keywords=["vLLM"],
    )
    score = score_case("order", _result(), expect)
    assert score.tools_ok is True

    bad = _result(tools=["update_evidence_notes", "search_knowledge_base"])
    score_bad = score_case("order_bad", bad, expect)
    assert score_bad.tools_ok is False
    assert score_bad.failure_class == "soft"


def test_hard_verified_without_sufficient():
    expect = CaseExpect(stop_reason="verified", baseline="multi")
    result = _result(sufficient=False)
    result["stop_reason"] = "verified"
    cls, _ = classify_failure(
        result=result,
        expect=expect,
        completed=True,
        tools_ok=True,
        trajectory_ok=True,
    )
    assert cls == "hard"


def test_cascading_soft_empty_retrieval():
    expect = CaseExpect(stop_reason="verified", baseline="multi")
    result = _result(
        answer="Secret teleportation feature exists.",
        evidence_items=[],
        kb_result="No relevant documents found in the knowledge base.",
        tools=["search_knowledge_base"],
        sufficient=True,
    )
    cls, notes = classify_failure(
        result=result,
        expect=expect,
        completed=True,
        tools_ok=True,
        trajectory_ok=True,
    )
    assert cls == "cascading_soft"
    assert "empty retrieval" in notes


def test_crash_is_hard():
    score = score_case("boom", None, CaseExpect(stop_reason="verified"), error="queue empty")
    assert score.failure_class == "hard"
    assert score.completed is False


def test_aggregate_rates():
    expect = CaseExpect(stop_reason="verified", answer_keywords=["vLLM"])
    scores = [
        score_case("a", _result(), expect),
        score_case("b", _result(answer="nope"), expect),
    ]
    metrics = aggregate(scores)
    assert metrics.n_cases == 2
    assert metrics.completion_rate == 0.5
    assert metrics.failure_counts["none"] == 1
    assert metrics.failure_counts["soft"] == 1


def test_harness_smoke_one_case():
    from eval.scripts import run_scenario_sync
    from eval.scoring import CaseExpect, score_case

    result, _, _ = run_scenario_sync(
        "clarification",
        "Compare things.",
    )
    score = score_case(
        "clarification",
        result,
        CaseExpect(
            stop_reason="clarification",
            required_tools=["ask_clarification"],
            answer_keywords=["Which docs"],
            expected_max_iterations=1,
        ),
    )
    assert score.completed is True
    assert score.failure_class == "none"
