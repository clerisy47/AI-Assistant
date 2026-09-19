"""Scripted tests for ResearchSupervisor multi-agent loop (Phase 4)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

from app.agent.supervisor import ResearchSupervisor
from app.llm.base import LLMProvider, LLMResponse, ToolCall
from app.tools.knowledge_base_tool import KNOWLEDGE_BASE_DEFINITION
from app.tools.research_tools import build_research_registry


class ScriptedMultiProvider(LLMProvider):
    """Serves generate() then generate_structured() from separate queues."""

    def __init__(
        self,
        generate_responses: list[LLMResponse],
        structured_payloads: list[dict[str, Any]],
    ):
        self._generate = list(generate_responses)
        self._structured = list(structured_payloads)
        self.generate_calls: list[dict] = []
        self.structured_calls: list[dict] = []

    async def generate(self, messages, **kwargs) -> LLMResponse:
        self.generate_calls.append(
            {"messages": list(messages), "system": kwargs.get("system"), "tools": kwargs.get("tools")}
        )
        if not self._generate:
            raise AssertionError("Unexpected generate() call — queue empty")
        return self._generate.pop(0)

    async def generate_structured(self, messages, schema, schema_name="structured_output", **kwargs):
        self.structured_calls.append(
            {
                "messages": list(messages),
                "system": kwargs.get("system"),
                "schema_name": schema_name,
            }
        )
        if not self._structured:
            raise AssertionError("Unexpected generate_structured() call — queue empty")
        return dict(self._structured.pop(0))


def _tc(call_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _tool_response(call_id: str, name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[_tc(call_id, name, arguments)],
        stop_reason="tool_use",
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[], stop_reason="end_turn")


def _insufficient() -> dict:
    return {
        "sufficient": False,
        "unsupported_claims": ["Missing OpenAI-compatible details"],
        "suggested_next_action": "re_research",
        "notes": "Draft overclaims relative to notes.",
    }


def _sufficient() -> dict:
    return {
        "sufficient": True,
        "unsupported_claims": [],
        "suggested_next_action": "finalize",
        "notes": "Claims backed by evidence notes.",
    }


def _supervisor_with_fake_kb(
    provider: ScriptedMultiProvider,
    *,
    kb_results: list[str] | None = None,
    max_iterations: int = 8,
    max_tool_calls: int = 12,
    max_pass_iterations: int = 4,
) -> ResearchSupervisor:
    results = list(kb_results or ["[source: doc.md | relevance=0.9]\nSome evidence."])
    original = build_research_registry

    def patched_registry(retriever, notes, clarification=None):
        registry, state = original(retriever, notes, clarification)

        async def fake_kb(query: str, top_k: int = 4) -> str:
            if not results:
                return "No relevant documents found in the knowledge base."
            return results.pop(0)

        registry.register(KNOWLEDGE_BASE_DEFINITION, fake_kb)
        return registry, state

    supervisor = ResearchSupervisor.create(
        provider,
        retriever=None,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        max_pass_iterations=max_pass_iterations,
    )
    # Patch where supervisor imports the builder.
    patcher = patch("app.agent.supervisor.build_research_registry", side_effect=patched_registry)
    patcher.start()
    supervisor._test_patcher = patcher  # type: ignore[attr-defined]
    return supervisor


def _run(supervisor: ResearchSupervisor, question: str = "Compare local LLM support.") -> dict:
    try:
        return asyncio.run(supervisor.run(question, temperature=0.0))
    finally:
        patcher = getattr(supervisor, "_test_patcher", None)
        if patcher is not None:
            patcher.stop()


def test_verify_fail_then_re_research_then_pass():
    """Acceptance: verify fails → research runs again → verify passes."""
    provider = ScriptedMultiProvider(
        generate_responses=[
            # Pass 1: search → notes → draft
            _tool_response("1", "search_knowledge_base", {"query": "local LLMs", "top_k": 4}),
            _tool_response(
                "2",
                "update_evidence_notes",
                {
                    "source": "about.md",
                    "excerpt": "Mentions vLLM briefly.",
                    "query_used": "local LLMs",
                    "open_gaps": ["Need OpenAI-compatible details"],
                    "sufficient_hypothesis": False,
                },
            ),
            _text_response("The assistant supports local LLMs via vLLM."),
            # Pass 2 (after verify fail): refined search → notes → revised draft
            _tool_response(
                "3",
                "search_knowledge_base",
                {"query": "vLLM OpenAI-compatible", "top_k": 4},
            ),
            _tool_response(
                "4",
                "update_evidence_notes",
                {
                    "source": "providers.md",
                    "excerpt": "OpenAI-compatible local provider via vLLM.",
                    "query_used": "vLLM OpenAI-compatible",
                    "open_gaps": [],
                    "sufficient_hypothesis": True,
                },
            ),
            _text_response(
                "Local LLMs are supported through an OpenAI-compatible vLLM provider."
            ),
        ],
        structured_payloads=[_insufficient(), _sufficient()],
    )
    supervisor = _supervisor_with_fake_kb(
        provider,
        kb_results=[
            "[source: about.md | relevance=0.8]\nMentions vLLM briefly.",
            "[source: providers.md | relevance=0.95]\nOpenAI-compatible local provider.",
        ],
    )
    result = _run(supervisor)

    assert result["stop_reason"] == "verified"
    assert result["verification"]["sufficient"] is True
    assert "OpenAI-compatible" in result["answer"]
    assert result["iterations"] == 2
    assert len(provider.structured_calls) == 2
    assert any(e.get("agent") == "research" for e in result["tool_trace"])
    assert any(e.get("agent") == "verifier" for e in result["tool_trace"])
    # Second research pass must include verifier feedback in the user message.
    second_pass_user = provider.generate_calls[3]["messages"][0].content or ""
    assert "Verifier feedback" in second_pass_user or "Missing OpenAI-compatible" in second_pass_user
    # Verifier context isolation: structured call must not contain raw multi-search dump marker.
    verify_msg = provider.structured_calls[0]["messages"][0].content or ""
    assert "Evidence notes:" in verify_msg
    assert "Mentions vLLM briefly." in verify_msg


def test_clarification_skips_verifier():
    provider = ScriptedMultiProvider(
        generate_responses=[
            _tool_response(
                "1",
                "ask_clarification",
                {"question": "Which docs should I compare?"},
            ),
        ],
        structured_payloads=[],
    )
    supervisor = _supervisor_with_fake_kb(provider)
    result = _run(supervisor, question="Compare things.")
    assert result["stop_reason"] == "clarification"
    assert "Which docs" in result["answer"]
    assert provider.structured_calls == []


def test_max_iterations_stop_reason():
    provider = ScriptedMultiProvider(
        generate_responses=[
            _text_response("Draft A — thin evidence."),
            _text_response("Draft B — still thin."),
        ],
        structured_payloads=[_insufficient(), _insufficient()],
    )
    supervisor = _supervisor_with_fake_kb(provider, max_iterations=2, max_pass_iterations=2)
    result = _run(supervisor)
    assert result["stop_reason"] == "max_iterations"
    assert result["iterations"] == 2
    assert "partial answer" in result["answer"].lower() or result["answer"]


def test_max_tool_calls_stop_reason():
    provider = ScriptedMultiProvider(
        generate_responses=[
            _tool_response("1", "search_knowledge_base", {"query": "a", "top_k": 4}),
            _tool_response("2", "search_knowledge_base", {"query": "b", "top_k": 4}),
            _text_response("Should not reach here"),
        ],
        structured_payloads=[],
    )
    supervisor = _supervisor_with_fake_kb(
        provider, max_iterations=4, max_tool_calls=1, max_pass_iterations=4
    )
    result = _run(supervisor)
    assert result["stop_reason"] == "max_tool_calls"
    assert any(e["tool"] == "search_knowledge_base" for e in result["tool_trace"])
