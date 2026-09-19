"""Research-path-only tools: evidence notes updates and clarification.

Registered on the ResearchAgent ToolRegistry (Phase 3), not on `/chat`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from app.agent.context_budget import cap_tool_result
from app.agent.evidence_notes import EvidenceNotes
from app.llm.base import ToolDefinition
from app.rag.retriever import Retriever
from app.tools.builtin_tools import CALCULATOR_DEFINITION, DATETIME_DEFINITION, calculator, get_current_datetime
from app.tools.knowledge_base_tool import KNOWLEDGE_BASE_DEFINITION, make_knowledge_base_tool
from app.tools.registry import ToolRegistry
from app.tools.skill_tool import register_skill_tools


@dataclass
class ClarificationState:
    """Side channel: when set, the research loop stops with stop_reason=clarification."""

    question: str | None = None

    @property
    def requested(self) -> bool:
        return self.question is not None


UPDATE_EVIDENCE_NOTES_DEFINITION = ToolDefinition(
    name="update_evidence_notes",
    description=(
        "Write or update structured EvidenceNotes for the current question. "
        "Add one evidence item (source, excerpt, query_used) and/or set open_gaps "
        "and sufficient_hypothesis. Prefer short excerpts; do not dump full retrieval text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Source document/file name for a new evidence item.",
            },
            "excerpt": {
                "type": "string",
                "description": "Short capped excerpt supporting a claim.",
            },
            "query_used": {
                "type": "string",
                "description": "The search query that produced this evidence.",
            },
            "relevance": {
                "type": "number",
                "description": "Optional relevance score from retrieval.",
            },
            "open_gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Remaining questions or missing angles to cover.",
            },
            "sufficient_hypothesis": {
                "type": "boolean",
                "description": "True if current notes seem enough to draft an answer.",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
)


ASK_CLARIFICATION_DEFINITION = ToolDefinition(
    name="ask_clarification",
    description=(
        "Ask the user a clarifying question when the research request is ambiguous "
        "or underspecified. Ends the research turn; do not continue searching after this."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Clear question to ask the user.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
)


def make_update_evidence_notes_tool(notes: EvidenceNotes):
    """Bind `update_evidence_notes` to a mutable EvidenceNotes instance."""

    async def update_evidence_notes(
        source: str | None = None,
        excerpt: str | None = None,
        query_used: str | None = None,
        relevance: float | None = None,
        open_gaps: list[str] | None = None,
        sufficient_hypothesis: bool | None = None,
    ) -> str:
        added = False
        if source is not None or excerpt is not None or query_used is not None:
            if not (source and excerpt is not None and query_used):
                raise ValueError(
                    "To add an evidence item, provide source, excerpt, and query_used together."
                )
            notes.add_item(
                source=source,
                excerpt=excerpt,
                query_used=query_used,
                relevance=relevance,
            )
            added = True

        if open_gaps is not None:
            notes.open_gaps = list(open_gaps)
        if sufficient_hypothesis is not None:
            notes.sufficient_hypothesis = bool(sufficient_hypothesis)

        if not added and open_gaps is None and sufficient_hypothesis is None:
            raise ValueError(
                "Provide at least one of: evidence item fields, open_gaps, or sufficient_hypothesis."
            )

        summary = {
            "ok": True,
            "items_count": len(notes.items),
            "open_gaps": notes.open_gaps,
            "sufficient_hypothesis": notes.sufficient_hypothesis,
            "added_item": added,
        }
        return json.dumps(summary)

    return update_evidence_notes


def make_ask_clarification_tool(state: ClarificationState):
    """Bind `ask_clarification` to a ClarificationState side channel."""

    async def ask_clarification(question: str) -> str:
        cleaned = (question or "").strip()
        if not cleaned:
            raise ValueError("clarification question must be non-empty")
        state.question = cleaned
        return json.dumps({"ok": True, "clarification": cleaned})

    return ask_clarification


def make_capped_knowledge_base_tool(retriever: Optional[Retriever]):
    """Research-path KB search with TOOL_RESULT_MAX_CHARS applied to the payload."""

    inner = make_knowledge_base_tool(retriever)

    async def search_knowledge_base(query: str, top_k: int = 4) -> str:
        result = await inner(query=query, top_k=top_k)
        return cap_tool_result(result)

    return search_knowledge_base


def build_research_registry(
    retriever: Optional[Retriever],
    notes: EvidenceNotes,
    clarification: ClarificationState | None = None,
) -> tuple[ToolRegistry, ClarificationState]:
    """Build the research-agent ToolRegistry and return (registry, clarification_state).

    Shared by tests and (later) the Phase 4 supervisor. Does not mutate `/chat` tools.
    """
    state = clarification if clarification is not None else ClarificationState()
    registry = ToolRegistry()
    registry.register(CALCULATOR_DEFINITION, calculator)
    registry.register(DATETIME_DEFINITION, get_current_datetime)
    registry.register(KNOWLEDGE_BASE_DEFINITION, make_capped_knowledge_base_tool(retriever))
    register_skill_tools(registry)
    registry.register(UPDATE_EVIDENCE_NOTES_DEFINITION, make_update_evidence_notes_tool(notes))
    registry.register(ASK_CLARIFICATION_DEFINITION, make_ask_clarification_tool(state))
    return registry, state
