"""Scripted tests for the Phase 3 ResearchAgent loop."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from app.agent.evidence_notes import EvidenceNotes
from app.agent.research_agent import ResearchAgent
from app.llm.base import LLMProvider, LLMResponse, ToolCall
from app.tools.builtin_tools import CALCULATOR_DEFINITION, DATETIME_DEFINITION, calculator, get_current_datetime
from app.tools.knowledge_base_tool import KNOWLEDGE_BASE_DEFINITION, make_knowledge_base_tool
from app.tools.registry import ToolRegistry
from app.tools.research_tools import ClarificationState, build_research_registry
from app.tools.skill_tool import build_skills_system_prefix


class ScriptedProvider(LLMProvider):
    """Returns pre-scripted responses in order, one per call to `generate`."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, messages, **kwargs) -> LLMResponse:
        self.calls.append({"messages": list(messages), "system": kwargs.get("system"), "tools": kwargs.get("tools")})
        return self._responses.pop(0)

    async def generate_structured(self, *args, **kwargs):
        raise NotImplementedError("not used in these tests")


def _tc(call_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _tool_response(call_id: str, name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[_tc(call_id, name, arguments)],
        stop_reason="tool_use",
    )


def _text_response(text: str, *, usage: dict | None = None) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=usage or {},
    )


def test_token_usage_from_provider_usage():
    provider = ScriptedProvider(
        [
            _text_response(
                "Draft answer.",
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            )
        ]
    )
    agent = _agent_with_fake_kb(provider)
    result = asyncio.run(agent.run("Simple question?"))
    usage = result["token_usage"]
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 120
    assert usage.estimated is False


def _agent_with_fake_kb(
    provider: ScriptedProvider,
    *,
    question: str = "What does the assistant support?",
    kb_results: list[str] | None = None,
    max_iterations: int = 8,
    max_tool_calls: int = 12,
) -> ResearchAgent:
    """Build a ResearchAgent whose KB tool returns canned strings in order."""
    results = list(kb_results or ["[source: doc.md | relevance=0.9]\nSome evidence."])
    notes = EvidenceNotes(question=question)
    clarification = ClarificationState()
    registry, clarification = build_research_registry(None, notes, clarification)

    async def fake_kb(query: str, top_k: int = 4) -> str:
        if not results:
            return "No relevant documents found in the knowledge base."
        return results.pop(0)

    # Replace capped KB handler with deterministic fake (still research registry).
    registry.register(KNOWLEDGE_BASE_DEFINITION, fake_kb)

    return ResearchAgent(
        provider,
        registry,
        evidence_notes=notes,
        clarification=clarification,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
    )


def test_re_search_then_draft():
    """Insufficient first search → notes with gaps → refined second search → draft."""
    provider = ScriptedProvider(
        [
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
            _tool_response(
                "3",
                "search_knowledge_base",
                {"query": "vLLM OpenAI-compatible local provider", "top_k": 4},
            ),
            _tool_response(
                "4",
                "update_evidence_notes",
                {
                    "source": "about.md",
                    "excerpt": "Supports local vLLM via OpenAI-compatible API.",
                    "query_used": "vLLM OpenAI-compatible local provider",
                    "open_gaps": [],
                    "sufficient_hypothesis": True,
                },
            ),
            _text_response("The assistant supports local vLLM via an OpenAI-compatible API."),
        ]
    )
    agent = _agent_with_fake_kb(
        provider,
        kb_results=[
            "[source: about.md | relevance=0.5]\nMentions vLLM briefly.",
            "[source: about.md | relevance=0.95]\nSupports local vLLM via OpenAI-compatible API.",
        ],
    )

    result = asyncio.run(agent.run("What local LLM support exists?"))

    assert result["stop_reason"] == "final"
    assert result["iterations"] > 1
    assert "vLLM" in result["draft_answer"]
    usage = result["token_usage"]
    assert usage is not None
    assert usage.total_tokens > 0
    assert usage.estimated is True  # scripted provider has empty usage
    kb_calls = [t for t in result["tool_trace"] if t["tool"] == "search_knowledge_base"]
    assert len(kb_calls) >= 2
    assert kb_calls[0]["arguments"]["query"] != kb_calls[1]["arguments"]["query"]
    notes = result["evidence_notes"]
    assert len(notes.items) >= 1
    assert notes.sufficient_hypothesis is True


def test_stops_at_max_iterations():
    always = _tool_response("x", "search_knowledge_base", {"query": "loop", "top_k": 1})
    provider = ScriptedProvider([always] * 20)
    agent = _agent_with_fake_kb(provider, max_iterations=3, max_tool_calls=100)

    result = asyncio.run(agent.run("loop forever"))

    assert result["stop_reason"] == "max_iterations"
    assert result["iterations"] == 3
    assert len(provider.calls) == 3
    assert "partial" in result["draft_answer"].lower() or "allowed" in result["draft_answer"].lower()


def test_stops_at_max_tool_calls():
    always = _tool_response("x", "search_knowledge_base", {"query": "loop", "top_k": 1})
    provider = ScriptedProvider([always] * 20)
    agent = _agent_with_fake_kb(provider, max_iterations=20, max_tool_calls=2)

    result = asyncio.run(agent.run("loop forever"))

    assert result["stop_reason"] == "max_tool_calls"
    kb_calls = [t for t in result["tool_trace"] if t["tool"] == "search_knowledge_base"]
    assert len(kb_calls) == 2
    assert len(result["tool_trace"]) == 2


def test_ask_clarification_stops():
    provider = ScriptedProvider(
        [
            _tool_response(
                "1",
                "ask_clarification",
                {"question": "Which documents should I compare?"},
            ),
        ]
    )
    agent = _agent_with_fake_kb(provider)

    result = asyncio.run(agent.run("Compare the things"))

    assert result["stop_reason"] == "clarification"
    assert "Which documents" in result["draft_answer"]
    assert result["iterations"] == 1
    assert len(result["tool_trace"]) == 1
    assert result["tool_trace"][0]["tool"] == "ask_clarification"


def test_skill_progressive_disclosure_in_research_loop():
    provider = ScriptedProvider(
        [
            _tool_response("1", "load_skill", {"name": "verified_research"}),
            _text_response("Loaded skill; ready to research."),
        ]
    )
    agent = _agent_with_fake_kb(provider)

    result = asyncio.run(agent.run("Research local LLMs"))

    # Catalog-only in system prompt before/while loading.
    first_system = provider.calls[0]["system"]
    assert "verified_research" in first_system
    assert "Do not self-declare" not in first_system
    assert "Anti-patterns" not in first_system
    # Prefix helper matches what agent uses.
    assert build_skills_system_prefix() in first_system

    assert result["stop_reason"] == "final"
    load_trace = result["tool_trace"][0]
    assert load_trace["tool"] == "load_skill"
    assert load_trace["is_error"] is False
    assert "Do not self-declare" in load_trace["result"] or "Anti-patterns" in load_trace["result"]
    assert "Handoff" in load_trace["result"]


def test_chat_registry_still_excludes_load_skill():
    """Isolation: classic /chat tool set must not include research-only tools."""
    registry = ToolRegistry()
    registry.register(CALCULATOR_DEFINITION, calculator)
    registry.register(DATETIME_DEFINITION, get_current_datetime)
    registry.register(KNOWLEDGE_BASE_DEFINITION, make_knowledge_base_tool(MagicMock()))

    names = {d.name for d in registry.definitions()}
    assert "load_skill" not in names
    assert "update_evidence_notes" not in names
    assert "ask_clarification" not in names
    assert "search_knowledge_base" in names


def test_research_registry_includes_research_tools():
    notes = EvidenceNotes(question="q")
    registry, _ = build_research_registry(None, notes)
    names = {d.name for d in registry.definitions()}
    assert "load_skill" in names
    assert "update_evidence_notes" in names
    assert "ask_clarification" in names
    assert "search_knowledge_base" in names
    assert "calculator" in names
