"""Constructs the configured LLMProvider.

This is the one place that knows how to turn `LLM_PROVIDER=...` into a live
client. Everything downstream (the agent orchestrator, the RAG query
endpoint, the structured-output endpoint) just calls `get_llm_provider()`
and uses the returned object through the common `LLMProvider` interface --
it never imports a concrete provider class directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from app.config import settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.openai_compatible_provider import OpenAICompatibleProvider


@lru_cache(maxsize=8)
def _build(provider_name: str) -> LLMProvider:
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=settings.ANTHROPIC_API_KEY, model=settings.ANTHROPIC_MODEL)
    if provider_name == "openai":
        return OpenAICompatibleProvider(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)
    if provider_name == "local":
        # Same client class as "openai" -- only the base URL changes. This is
        # what lets a self-hosted vLLM server be a first-class, swappable
        # backend instead of a bolted-on special case.
        return OpenAICompatibleProvider(api_key="not-needed", model=settings.VLLM_MODEL, base_url=settings.VLLM_BASE_URL)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider_name}'. Expected one of: anthropic, openai, local.")


def get_llm_provider(name: Optional[str] = None) -> LLMProvider:
    """Get the LLM provider for `name`, or the configured default if omitted.

    Clients (e.g. a `/chat` request) may pass a `provider` override to talk
    to a *different* backend for a single request without restarting the
    service -- handy for side-by-side comparisons between, say, Claude and a
    locally hosted Llama 3.
    """
    return _build((name or settings.LLM_PROVIDER).lower())
