"""Unit tests for token usage helpers (Phase 5)."""

from __future__ import annotations

from app.llm.base import LLMMessage, LLMResponse
from app.llm.usage import (
    TokenUsage,
    estimate_tokens,
    normalize_usage,
    usage_from_generate,
)


def test_normalize_maps_vendor_keys():
    usage = normalize_usage({"input_tokens": 10, "output_tokens": 3})
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 3
    assert usage.total_tokens == 13


def test_normalize_prompt_completion_keys():
    usage = normalize_usage({"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10})
    assert usage.to_dict() == {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
    }


def test_estimate_tokens_chars_div_4():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_usage_from_generate_estimates_when_empty():
    messages = [LLMMessage(role="user", content="Hello world question")]
    response = LLMResponse(content="Answer text here", tool_calls=[], stop_reason="end_turn")
    usage = usage_from_generate(response, system="sys", messages=messages)
    assert usage.estimated is True
    assert usage.total_tokens > 0


def test_token_usage_add_with_agent():
    a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = TokenUsage(prompt_tokens=20, completion_tokens=4, total_tokens=24)
    summed = a.add(b, agent="research")
    assert summed.prompt_tokens == 30
    assert summed.completion_tokens == 9
    assert summed.total_tokens == 39
    assert summed.by_agent == {"research": 24}
