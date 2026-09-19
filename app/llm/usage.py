"""Token usage normalization and estimation (Phase 5).

Providers return vendor-specific usage keys. We normalize to
``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.

When an API omits usage (scripted providers, some local servers), we
estimate with a chars÷4 heuristic — a common rough tokenizer proxy that
needs no extra dependency. Mark such totals with ``estimated=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from app.llm.base import LLMMessage, LLMResponse


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)
    estimated: bool = False

    def add(self, other: "TokenUsage", *, agent: str | None = None) -> "TokenUsage":
        """Return a new TokenUsage that is the sum of self and other."""
        by_agent = dict(self.by_agent)
        other_total = other.total_tokens
        if agent:
            by_agent[agent] = by_agent.get(agent, 0) + other_total
        else:
            for name, amount in other.by_agent.items():
                by_agent[name] = by_agent.get(name, 0) + amount
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            by_agent=by_agent,
            estimated=self.estimated or other.estimated,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.by_agent:
            out["by_agent"] = dict(self.by_agent)
        if self.estimated:
            out["estimated"] = True
        return out


def estimate_tokens(text: str) -> int:
    """Rough token count: max(1, len(text) // 4) for non-empty text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def normalize_usage(raw: Optional[Mapping[str, Any]]) -> TokenUsage:
    """Map provider usage dicts onto the canonical TokenUsage shape."""
    if not raw:
        return TokenUsage()

    prompt = raw.get("prompt_tokens")
    if prompt is None:
        prompt = raw.get("input_tokens", 0)
    completion = raw.get("completion_tokens")
    if completion is None:
        completion = raw.get("output_tokens", 0)

    prompt_i = int(prompt or 0)
    completion_i = int(completion or 0)
    total = raw.get("total_tokens")
    total_i = int(total) if total is not None else prompt_i + completion_i

    by_agent: dict[str, int] = {}
    raw_by = raw.get("by_agent")
    if isinstance(raw_by, Mapping):
        by_agent = {str(k): int(v) for k, v in raw_by.items()}

    estimated = bool(raw.get("estimated", False))
    return TokenUsage(
        prompt_tokens=prompt_i,
        completion_tokens=completion_i,
        total_tokens=total_i,
        by_agent=by_agent,
        estimated=estimated,
    )


def _message_text(message: LLMMessage) -> str:
    parts: list[str] = []
    if message.content:
        parts.append(message.content)
    for call in message.tool_calls or []:
        parts.append(call.name)
        parts.append(str(call.arguments))
    return "\n".join(parts)


def estimate_turn_usage(
    *,
    system: Optional[str],
    messages: list[LLMMessage],
    response: LLMResponse,
) -> TokenUsage:
    """Estimate prompt + completion tokens for one generate() turn."""
    prompt_chunks = [system or ""]
    prompt_chunks.extend(_message_text(m) for m in messages)
    prompt_tokens = estimate_tokens("\n".join(prompt_chunks))

    completion_chunks = [response.content or ""]
    for call in response.tool_calls or []:
        completion_chunks.append(call.name)
        completion_chunks.append(str(call.arguments))
    completion_tokens = estimate_tokens("\n".join(completion_chunks))

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated=True,
    )


def usage_from_generate(
    response: LLMResponse,
    *,
    system: Optional[str],
    messages: list[LLMMessage],
) -> TokenUsage:
    """Prefer API usage; fall back to chars÷4 estimation when missing."""
    normalized = normalize_usage(response.usage)
    if normalized.total_tokens > 0 or (
        normalized.prompt_tokens > 0 or normalized.completion_tokens > 0
    ):
        return normalized
    return estimate_turn_usage(system=system, messages=messages, response=response)


def estimate_structured_usage(
    *,
    system: Optional[str],
    messages: list[LLMMessage],
    data: Mapping[str, Any],
) -> TokenUsage:
    """Estimate usage for a structured-output call when the API omits it."""
    prompt_chunks = [system or ""]
    prompt_chunks.extend(_message_text(m) for m in messages)
    prompt_tokens = estimate_tokens("\n".join(prompt_chunks))
    completion_tokens = estimate_tokens(str(dict(data)))
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        estimated=True,
    )
