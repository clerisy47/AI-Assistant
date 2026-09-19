"""Structured external evidence notes for verified research.

Material findings live here instead of only in ephemeral tool messages, so the
verifier can see draft + notes without the full exploratory retrieval dump.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.context_budget import cap_excerpt


class EvidenceItem(BaseModel):
    source: str
    excerpt: str
    query_used: str
    relevance: float | None = None


class EvidenceNotes(BaseModel):
    question: str
    items: list[EvidenceItem] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)
    sufficient_hypothesis: bool = False

    def add_item(
        self,
        *,
        source: str,
        excerpt: str,
        query_used: str,
        relevance: float | None = None,
        max_excerpt_chars: int | None = None,
    ) -> EvidenceItem:
        """Append a capped evidence item and return it."""
        item = EvidenceItem(
            source=source,
            excerpt=cap_excerpt(excerpt, max_excerpt_chars),
            query_used=query_used,
            relevance=relevance,
        )
        self.items.append(item)
        return item


def to_verifier_context(notes: EvidenceNotes) -> str:
    """Compact, stable serialization for the verifier (no raw multi-search dumps)."""
    lines: list[str] = [
        f"Question: {notes.question}",
        f"Sufficient hypothesis: {notes.sufficient_hypothesis}",
    ]
    if notes.open_gaps:
        lines.append("Open gaps:")
        for gap in notes.open_gaps:
            lines.append(f"  - {gap}")
    else:
        lines.append("Open gaps: (none)")

    if not notes.items:
        lines.append("Evidence items: (none)")
    else:
        lines.append(f"Evidence items ({len(notes.items)}):")
        for i, item in enumerate(notes.items, start=1):
            relevance = f"{item.relevance:.3f}" if item.relevance is not None else "n/a"
            lines.append(
                f"  [{i}] source={item.source} | relevance={relevance} | query={item.query_used}"
            )
            lines.append(f"      excerpt: {item.excerpt}")

    return "\n".join(lines)
