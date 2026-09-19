"""Helpers that keep research-loop context from saturating.

Multi-turn retrieval dumps and long excerpts quickly crowd the window and
degrade later tool choice. Cap raw tool payloads and optionally replace older
tool results with short placeholders once structured evidence notes exist.
"""

from __future__ import annotations

from typing import Sequence

from app.config import settings
from app.llm.base import LLMMessage

_TRUNCATION_SUFFIX = "...[truncated]"
COMPACTED_PLACEHOLDER = "[tool result compacted — see evidence notes]"


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_SUFFIX):
        return _TRUNCATION_SUFFIX[:max_chars]
    keep = max_chars - len(_TRUNCATION_SUFFIX)
    return text[:keep] + _TRUNCATION_SUFFIX


def cap_tool_result(text: str, max_chars: int | None = None) -> str:
    """Truncate a raw tool result for inclusion in the research conversation."""
    limit = settings.TOOL_RESULT_MAX_CHARS if max_chars is None else max_chars
    return _truncate(text or "", limit)


def cap_excerpt(text: str, max_chars: int | None = None) -> str:
    """Truncate an evidence excerpt for structured notes / verifier context."""
    limit = settings.EVIDENCE_EXCERPT_MAX_CHARS if max_chars is None else max_chars
    return _truncate(text or "", limit)


def compact_messages(
    messages: Sequence[LLMMessage],
    keep_last_n_tool_results: int,
) -> list[LLMMessage]:
    """Return a copy of `messages` with older tool payloads compacted.

    The last `keep_last_n_tool_results` messages with `role == "tool"` keep
    their content; earlier tool messages are replaced with a short placeholder.
    Non-tool messages are left unchanged. Does not mutate the input sequence.
    """
    if keep_last_n_tool_results < 0:
        raise ValueError("keep_last_n_tool_results must be >= 0")

    tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
    keep_set = set(tool_indices[-keep_last_n_tool_results:]) if keep_last_n_tool_results else set()

    compacted: list[LLMMessage] = []
    for i, msg in enumerate(messages):
        if msg.role == "tool" and i not in keep_set:
            compacted.append(
                LLMMessage(
                    role="tool",
                    content=COMPACTED_PLACEHOLDER,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    is_error=msg.is_error,
                )
            )
        else:
            compacted.append(msg)
    return compacted
