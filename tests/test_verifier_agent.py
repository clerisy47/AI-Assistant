"""Scripted tests for VerifierAgent context isolation (Phase 4–5)."""

from __future__ import annotations

import asyncio

from app.agent.evidence_notes import EvidenceNotes
from app.agent.verifier_agent import VerificationResult, VerifierAgent
from app.llm.base import LLMProvider, StructuredLLMResponse


class RecordingStructuredProvider(LLMProvider):
    def __init__(self, payload: dict, *, usage: dict | None = None):
        self._payload = payload
        self._usage = usage or {}
        self.calls: list[dict] = []

    async def generate(self, messages, **kwargs):
        raise NotImplementedError("verifier uses generate_structured")

    async def generate_structured(self, messages, schema, schema_name="structured_output", **kwargs):
        self.calls.append(
            {
                "messages": list(messages),
                "system": kwargs.get("system"),
                "schema_name": schema_name,
                "schema": schema,
            }
        )
        return StructuredLLMResponse(data=dict(self._payload), usage=dict(self._usage))


def test_verifier_context_is_draft_plus_notes_only():
    notes = EvidenceNotes(question="What about local LLMs?")
    notes.add_item(
        source="about.md",
        excerpt="Supports vLLM via OpenAI-compatible API.",
        query_used="local LLM",
        relevance=0.9,
    )
    provider = RecordingStructuredProvider(
        {
            "sufficient": True,
            "unsupported_claims": [],
            "suggested_next_action": "finalize",
            "notes": "Supported.",
        },
        usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    )
    agent = VerifierAgent(provider)
    draft = "The assistant supports local LLMs via vLLM."
    raw_dump = "RAW_SEARCH_DUMP " + ("x" * 500)

    result, trace, usage = asyncio.run(
        agent.verify(question=notes.question, draft_answer=draft, evidence_notes=notes)
    )

    assert result.sufficient is True
    assert result.suggested_next_action == "finalize"
    assert trace["agent"] == "verifier"
    assert trace["tool"] == "verify_claims"
    assert usage.total_tokens == 60
    assert trace["token_usage"]["total_tokens"] == 60
    assert len(provider.calls) == 1

    user_content = provider.calls[0]["messages"][0].content or ""
    assert draft in user_content
    assert "Supports vLLM via OpenAI-compatible API." in user_content
    assert "about.md" in user_content
    assert raw_dump not in user_content
    assert "search_knowledge_base" not in user_content


def test_verifier_normalizes_sufficient_to_finalize():
    provider = RecordingStructuredProvider(
        {
            "sufficient": True,
            "unsupported_claims": [],
            "suggested_next_action": "re_research",
            "notes": "ok",
        }
    )
    notes = EvidenceNotes(question="q")
    result, _, usage = asyncio.run(
        VerifierAgent(provider).verify(question="q", draft_answer="a", evidence_notes=notes)
    )
    assert result.suggested_next_action == "finalize"
    assert usage.estimated is True  # empty provider usage → estimate


def test_verifier_fail_closed_on_structured_error():
    class BoomProvider(LLMProvider):
        async def generate(self, *args, **kwargs):
            raise NotImplementedError

        async def generate_structured(self, *args, **kwargs):
            raise RuntimeError("provider down")

    notes = EvidenceNotes(question="q")
    result, trace, usage = asyncio.run(
        VerifierAgent(BoomProvider()).verify(
            question="q", draft_answer="draft", evidence_notes=notes
        )
    )
    assert result.sufficient is False
    assert result.suggested_next_action == "re_research"
    assert trace["is_error"] is True
    assert isinstance(result, VerificationResult)
    assert usage.total_tokens == 0
