"""Unit tests for evidence notes and context-budget helpers."""

from app.agent.context_budget import (
    COMPACTED_PLACEHOLDER,
    cap_excerpt,
    cap_tool_result,
    compact_messages,
)
from app.agent.evidence_notes import EvidenceNotes, to_verifier_context
from app.config import settings
from app.llm.base import LLMMessage


def test_evidence_notes_json_round_trip():
    notes = EvidenceNotes(
        question="What does local LLM support look like?",
        open_gaps=["need vLLM details"],
        sufficient_hypothesis=False,
    )
    notes.add_item(
        source="about_this_assistant.md",
        excerpt="Supports local vLLM via OpenAI-compatible provider.",
        query_used="local LLM vLLM",
        relevance=0.91,
    )

    restored = EvidenceNotes.model_validate_json(notes.model_dump_json())

    assert restored.question == notes.question
    assert restored.open_gaps == notes.open_gaps
    assert restored.sufficient_hypothesis is False
    assert len(restored.items) == 1
    assert restored.items[0].source == "about_this_assistant.md"
    assert restored.items[0].relevance == 0.91


def test_add_item_caps_excerpt_length():
    notes = EvidenceNotes(question="q")
    long_excerpt = "x" * (settings.EVIDENCE_EXCERPT_MAX_CHARS + 200)

    item = notes.add_item(
        source="doc.md",
        excerpt=long_excerpt,
        query_used="q",
        relevance=0.5,
    )

    assert len(item.excerpt) <= settings.EVIDENCE_EXCERPT_MAX_CHARS
    assert item.excerpt.endswith("...[truncated]")
    assert len(notes.items) == 1


def test_cap_tool_result_bounds_and_preserves_short():
    short = "already short"
    assert cap_tool_result(short) == short
    assert cap_tool_result("") == ""

    long = "y" * (settings.TOOL_RESULT_MAX_CHARS + 500)
    capped = cap_tool_result(long)
    assert len(capped) <= settings.TOOL_RESULT_MAX_CHARS
    assert capped.endswith("...[truncated]")

    custom = cap_tool_result("abcdef", max_chars=5)
    assert len(custom) <= 5
    assert custom.endswith("...[truncated]") or len(custom) <= 5


def test_cap_excerpt_respects_explicit_max():
    text = "abcdefghij"
    assert cap_excerpt(text, max_chars=100) == text
    capped = cap_excerpt(text, max_chars=8)
    assert len(capped) <= 8
    assert "...[truncated]" in capped or len(capped) <= 8


def test_compact_messages_keeps_last_n_tool_results():
    messages = [
        LLMMessage(role="user", content="q"),
        LLMMessage(role="assistant", content=None, tool_calls=[]),
        LLMMessage(role="tool", content="result-1", tool_call_id="c1", name="search_knowledge_base"),
        LLMMessage(role="assistant", content=None, tool_calls=[]),
        LLMMessage(role="tool", content="result-2", tool_call_id="c2", name="search_knowledge_base"),
        LLMMessage(role="assistant", content=None, tool_calls=[]),
        LLMMessage(role="tool", content="result-3", tool_call_id="c3", name="search_knowledge_base"),
    ]

    compacted = compact_messages(messages, keep_last_n_tool_results=2)

    tool_contents = [m.content for m in compacted if m.role == "tool"]
    assert tool_contents == [COMPACTED_PLACEHOLDER, "result-2", "result-3"]
    assert compacted[0].content == "q"
    # Originals unchanged
    assert messages[2].content == "result-1"


def test_compact_messages_keep_zero_compacts_all_tools():
    messages = [
        LLMMessage(role="tool", content="a", tool_call_id="1", name="t"),
        LLMMessage(role="tool", content="b", tool_call_id="2", name="t"),
    ]
    compacted = compact_messages(messages, keep_last_n_tool_results=0)
    assert all(m.content == COMPACTED_PLACEHOLDER for m in compacted)


def test_to_verifier_context_includes_sources_and_stays_bounded():
    notes = EvidenceNotes(question="Compare calculator vs KB guidance", open_gaps=["scope"])
    notes.add_item(
        source="guide.md",
        excerpt="Use calculator for arithmetic.",
        query_used="calculator guidance",
        relevance=0.8,
    )
    notes.add_item(
        source="rag.md",
        excerpt="x" * 2000,  # will be capped on add
        query_used="knowledge base",
    )

    ctx = to_verifier_context(notes)

    assert "Compare calculator vs KB guidance" in ctx
    assert "guide.md" in ctx
    assert "rag.md" in ctx
    assert "Use calculator for arithmetic." in ctx
    assert "Open gaps:" in ctx
    # Verifier context must not contain the uncapped 2000-char dump
    assert "x" * 2000 not in ctx
    assert len(ctx) < 5000
