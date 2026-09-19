"""Scripted ResearchSupervisor scenarios for the eval harness (no live LLM/Qdrant)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import patch

from app.agent.supervisor import ResearchSupervisor
from app.llm.base import LLMProvider, LLMResponse, StructuredLLMResponse, ToolCall
from app.tools.knowledge_base_tool import KNOWLEDGE_BASE_DEFINITION
from app.tools.research_tools import build_research_registry


class ScriptedMultiProvider(LLMProvider):
    """Serves generate() then generate_structured() from separate queues."""

    def __init__(
        self,
        generate_responses: list[LLMResponse],
        structured_payloads: list[dict[str, Any]],
        *,
        structured_usage: dict[str, int] | None = None,
    ):
        self._generate = list(generate_responses)
        self._structured = list(structured_payloads)
        self._structured_usage = structured_usage or {
            "prompt_tokens": 40,
            "completion_tokens": 10,
            "total_tokens": 50,
        }
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
        return StructuredLLMResponse(
            data=dict(self._structured.pop(0)),
            usage=dict(self._structured_usage),
        )


def _tc(call_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def tool_response(call_id: str, name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[_tc(call_id, name, arguments)],
        stop_reason="tool_use",
        usage=_usage(30, 5),
    )


def text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=_usage(25, 15),
    )


def insufficient() -> dict:
    return {
        "sufficient": False,
        "unsupported_claims": ["Missing OpenAI-compatible details"],
        "suggested_next_action": "re_research",
        "notes": "Draft overclaims relative to notes.",
    }


def sufficient() -> dict:
    return {
        "sufficient": True,
        "unsupported_claims": [],
        "suggested_next_action": "finalize",
        "notes": "Claims backed by evidence notes.",
    }


def supervisor_with_fake_kb(
    provider: ScriptedMultiProvider,
    *,
    kb_results: list[str] | None = None,
    max_iterations: int = 8,
    max_tool_calls: int = 12,
    max_pass_iterations: int = 4,
) -> ResearchSupervisor:
    results = list(kb_results) if kb_results is not None else [
        "[source: doc.md | relevance=0.9]\nSome evidence."
    ]
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
    patcher = patch("app.agent.supervisor.build_research_registry", side_effect=patched_registry)
    patcher.start()
    supervisor._eval_patcher = patcher  # type: ignore[attr-defined]
    return supervisor


def stop_patcher(supervisor: ResearchSupervisor) -> None:
    patcher = getattr(supervisor, "_eval_patcher", None)
    if patcher is not None:
        patcher.stop()


@dataclass
class ScenarioBundle:
    """Ready-to-run supervisor plus optional second provider for baseline compare."""

    supervisor: ResearchSupervisor
    baseline: str = "multi"
    # For compare_baseline: build a fresh single-agent supervisor
    single_factory: Callable[[], ResearchSupervisor] | None = None


def _search_notes_draft(
    *,
    search_id: str,
    notes_id: str,
    query: str,
    source: str,
    excerpt: str,
    draft: str,
    open_gaps: list[str] | None = None,
    sufficient_hypothesis: bool = True,
) -> list[LLMResponse]:
    return [
        tool_response(search_id, "search_knowledge_base", {"query": query, "top_k": 4}),
        tool_response(
            notes_id,
            "update_evidence_notes",
            {
                "source": source,
                "excerpt": excerpt,
                "query_used": query,
                "open_gaps": open_gaps or [],
                "sufficient_hypothesis": sufficient_hypothesis,
            },
        ),
        text_response(draft),
    ]


def scenario_sufficient_first_search() -> ScenarioBundle:
    provider = ScriptedMultiProvider(
        generate_responses=_search_notes_draft(
            search_id="1",
            notes_id="2",
            query="local LLMs",
            source="about.md",
            excerpt="Local LLMs via OpenAI-compatible vLLM provider.",
            draft="The assistant supports local LLMs through an OpenAI-compatible vLLM provider.",
        ),
        structured_payloads=[sufficient()],
    )
    return ScenarioBundle(
        supervisor_with_fake_kb(
            provider,
            kb_results=[
                "[source: about.md | relevance=0.95]\nLocal LLMs via OpenAI-compatible vLLM."
            ],
        )
    )


def scenario_needs_re_search() -> ScenarioBundle:
    provider = ScriptedMultiProvider(
        generate_responses=[
            *_search_notes_draft(
                search_id="1",
                notes_id="2",
                query="local LLMs",
                source="about.md",
                excerpt="Mentions vLLM briefly.",
                draft="The assistant supports local LLMs via vLLM.",
                open_gaps=["Need OpenAI-compatible details"],
                sufficient_hypothesis=False,
            ),
            *_search_notes_draft(
                search_id="3",
                notes_id="4",
                query="vLLM OpenAI-compatible",
                source="providers.md",
                excerpt="OpenAI-compatible local provider via vLLM.",
                draft="Local LLMs are supported through an OpenAI-compatible vLLM provider.",
            ),
        ],
        structured_payloads=[insufficient(), sufficient()],
    )
    return ScenarioBundle(
        supervisor_with_fake_kb(
            provider,
            kb_results=[
                "[source: about.md | relevance=0.8]\nMentions vLLM briefly.",
                "[source: providers.md | relevance=0.95]\nOpenAI-compatible local provider.",
            ],
        )
    )


def scenario_clarification() -> ScenarioBundle:
    provider = ScriptedMultiProvider(
        generate_responses=[
            tool_response(
                "1",
                "ask_clarification",
                {"question": "Which docs should I compare?"},
            ),
        ],
        structured_payloads=[],
    )
    return ScenarioBundle(supervisor_with_fake_kb(provider))


def scenario_verify_fail_then_revise() -> ScenarioBundle:
    # Same branching as needs_re_search; rubric keywords emphasize the revision.
    return scenario_needs_re_search()


def scenario_max_iterations_stop() -> ScenarioBundle:
    provider = ScriptedMultiProvider(
        generate_responses=[
            text_response("Draft A — thin evidence."),
            text_response("Draft B — still thin."),
        ],
        structured_payloads=[insufficient(), insufficient()],
    )
    return ScenarioBundle(
        supervisor_with_fake_kb(provider, max_iterations=2, max_pass_iterations=2)
    )


def scenario_max_tool_calls_stop() -> ScenarioBundle:
    provider = ScriptedMultiProvider(
        generate_responses=[
            tool_response("1", "search_knowledge_base", {"query": "a", "top_k": 4}),
            tool_response("2", "search_knowledge_base", {"query": "b", "top_k": 4}),
            text_response("Should not reach here"),
        ],
        structured_payloads=[],
    )
    return ScenarioBundle(
        supervisor_with_fake_kb(
            provider, max_iterations=4, max_tool_calls=1, max_pass_iterations=4
        )
    )


def scenario_tool_order_search_then_notes() -> ScenarioBundle:
    return scenario_sufficient_first_search()


def scenario_weak_answer_soft() -> ScenarioBundle:
    """Verified path, but draft omits a strong keyword the rubric expects."""
    provider = ScriptedMultiProvider(
        generate_responses=_search_notes_draft(
            search_id="1",
            notes_id="2",
            query="local LLMs",
            source="about.md",
            excerpt="Mentions vLLM.",
            draft="Local models are supported somehow.",
        ),
        structured_payloads=[sufficient()],
    )
    return ScenarioBundle(
        supervisor_with_fake_kb(
            provider,
            kb_results=["[source: about.md | relevance=0.9]\nMentions vLLM."],
        )
    )


def scenario_baseline_token_compare() -> ScenarioBundle:
    def _make_provider() -> ScriptedMultiProvider:
        return ScriptedMultiProvider(
            generate_responses=_search_notes_draft(
                search_id="1",
                notes_id="2",
                query="local",
                source="about.md",
                excerpt="vLLM supported.",
                draft="Local LLMs via vLLM.",
            ),
            structured_payloads=[sufficient()],
        )

    multi = supervisor_with_fake_kb(
        _make_provider(),
        kb_results=["[source: about.md | relevance=0.9]\nvLLM supported."],
    )

    def single_factory() -> ResearchSupervisor:
        # Single baseline never calls verifier; only generate queue needed.
        provider = ScriptedMultiProvider(
            generate_responses=_search_notes_draft(
                search_id="1",
                notes_id="2",
                query="local",
                source="about.md",
                excerpt="vLLM supported.",
                draft="Local LLMs via vLLM.",
            ),
            structured_payloads=[],
        )
        return supervisor_with_fake_kb(
            provider,
            kb_results=["[source: about.md | relevance=0.9]\nvLLM supported."],
        )

    return ScenarioBundle(supervisor=multi, baseline="multi", single_factory=single_factory)


def scenario_empty_retrieval_cascade() -> ScenarioBundle:
    """Empty KB → confident draft → verifier rubber-stamps → cascading soft."""
    provider = ScriptedMultiProvider(
        generate_responses=[
            tool_response("1", "search_knowledge_base", {"query": "secret feature", "top_k": 4}),
            text_response(
                "The corpus clearly documents a secret feature that enables teleportation."
            ),
        ],
        structured_payloads=[sufficient()],
    )
    return ScenarioBundle(supervisor_with_fake_kb(provider, kb_results=[]))


SCENARIOS: dict[str, Callable[[], ScenarioBundle]] = {
    "sufficient_first_search": scenario_sufficient_first_search,
    "needs_re_search": scenario_needs_re_search,
    "clarification": scenario_clarification,
    "verify_fail_then_revise": scenario_verify_fail_then_revise,
    "max_iterations_stop": scenario_max_iterations_stop,
    "max_tool_calls_stop": scenario_max_tool_calls_stop,
    "tool_order_search_then_notes": scenario_tool_order_search_then_notes,
    "weak_answer_soft": scenario_weak_answer_soft,
    "baseline_token_compare": scenario_baseline_token_compare,
    "empty_retrieval_cascade": scenario_empty_retrieval_cascade,
}


async def run_supervisor(
    bundle: ScenarioBundle,
    question: str,
    *,
    baseline: str | None = None,
) -> dict[str, Any]:
    mode = baseline or bundle.baseline
    supervisor = bundle.supervisor
    try:
        return await supervisor.run(question, temperature=0.0, baseline=mode)  # type: ignore[arg-type]
    finally:
        stop_patcher(supervisor)


def run_scenario_sync(
    scenario_id: str,
    question: str,
    *,
    compare_baseline: bool = False,
) -> tuple[dict[str, Any], int | None, int | None]:
    """Return (primary_result, tokens_multi, tokens_single)."""
    factory = SCENARIOS.get(scenario_id)
    if factory is None:
        raise KeyError(f"Unknown scenario: {scenario_id}")

    bundle = factory()
    if compare_baseline and bundle.single_factory is not None:
        multi_result = asyncio.run(run_supervisor(bundle, question, baseline="multi"))
        single_bundle = ScenarioBundle(supervisor=bundle.single_factory(), baseline="single")
        single_result = asyncio.run(run_supervisor(single_bundle, question, baseline="single"))
        tokens_multi = int((multi_result.get("token_usage") or {}).get("total_tokens") or 0)
        tokens_single = int((single_result.get("token_usage") or {}).get("total_tokens") or 0)
        return multi_result, tokens_multi, tokens_single

    result = asyncio.run(run_supervisor(bundle, question))
    return result, None, None
