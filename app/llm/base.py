"""Provider-agnostic LLM abstraction.

Every concrete backend (Anthropic Claude, OpenAI, local vLLM) implements the
same `LLMProvider` interface. This is what lets the rest of the application
(the agent loop, the RAG endpoints, the structured-output endpoint) stay
completely unaware of which backend is actually answering a given request --
switching providers is a one-line config change (`LLM_PROVIDER=...`), not a
code change.

Design note: `generate_structured` is declared here but implemented per
provider rather than with a single shared implementation, because the
*right* way to guarantee schema-valid JSON differs by backend:

  - Anthropic:  native JSON outputs (`output_config.format`, constrained
                decoding) -- see app/llm/anthropic_provider.py
  - OpenAI/vLLM: native `response_format={"type": "json_schema", ...}`
                -- see app/llm/openai_compatible_provider.py

A provider that has *no* native structured-output support could still
implement this method with the older "force a synthetic tool call and
parse its arguments" trick -- both concrete providers here support the
real thing, so that fallback isn't needed, but it's a reasonable pattern
to know about if you add a fifth provider that lacks native support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolDefinition:
    """A tool the model is allowed to call, described as JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema: {"type": "object", "properties": {...}, ...}
    # When True, providers that support it (Anthropic strict tool use, OpenAI
    # strict function calling) guarantee the arguments the model returns
    # validate against `parameters` -- no more "expected int, got '3'" bugs.
    strict: bool = True


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMMessage:
    """One turn in a conversation, in our internal (provider-neutral) shape."""

    role: Role
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)  # role == "assistant"
    tool_call_id: Optional[str] = None  # role == "tool": which call this answers
    name: Optional[str] = None  # role == "tool": tool name (for providers that want it)
    is_error: bool = False  # role == "tool": did the tool raise/fail?


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class StructuredLLMResponse:
    """Result of ``generate_structured`` — parsed data plus token usage."""

    data: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON Schema (e.g. from Pydantic's `model_json_schema()`)
    into the flavor required by strict/constrained decoding across providers:
    every property listed in `required`, and `additionalProperties: False`.

    Only handles flat, single-level object schemas -- which covers the demo
    schemas in this project. Deeply nested models with `$defs`/`$ref` need
    the same treatment applied recursively; see the README's "Extending
    structured output" section.
    """
    schema = dict(schema)
    schema["additionalProperties"] = False
    if schema.get("type") == "object":
        schema["required"] = list(schema.get("properties", {}).keys())
    return schema


class LLMProvider(ABC):
    """Common interface implemented by every LLM backend."""

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        system: Optional[str] = None,
        tools: Optional[list[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,  # None/"auto" | "required" | a specific tool name
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Run one turn of generation, optionally with tools available."""
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        schema_name: str = "structured_output",
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> StructuredLLMResponse:
        """Return schema-valid JSON plus usage (estimated when the API omits it)."""
        raise NotImplementedError
