"""Anthropic Claude backend.

Uses two distinct native Claude API features, on purpose:

- Regular tool calling (`tools` + `tool_choice`) for the agent loop, where
  Claude decides *which* tool (if any) to call -- see `generate()`.
- Native JSON outputs (`output_config.format`, GA as of 2026, constrained
  decoding) for guaranteed-valid structured JSON -- see
  `generate_structured()`. This is not the older "define a fake tool and
  hope the model calls it" trick; the API compiles the schema and
  constrains generation token-by-token, so the result is *guaranteed*
  valid JSON, not just usually valid.

Docs: https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use
      https://platform.claude.com/docs/en/build-with-claude/structured-outputs
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from anthropic import AsyncAnthropic

from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall, ToolDefinition, strict_json_schema

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY is not set; requests to Anthropic will fail.")
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    # -- message translation -------------------------------------------------

    def _convert_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content or "",
                                "is_error": m.is_error,
                            }
                        ],
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append({"role": m.role, "content": m.content or ""})
        return converted

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": strict_json_schema(t.parameters) if t.strict else t.parameters,
                "strict": t.strict,
            }
            for t in tools
        ]

    # -- public API ------------------------------------------------------------

    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        system: Optional[str] = None,
        tools: Optional[list[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": self._convert_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            if tool_choice in (None, "auto"):
                kwargs["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                kwargs["tool_choice"] = {"type": "any"}
            else:
                kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        response = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        return LLMResponse(
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            raw=response,
        )

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        schema_name: str = "structured_output",
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": self._convert_messages(messages),
            "output_config": {"format": {"type": "json_schema", "schema": strict_json_schema(schema)}},
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON despite structured output mode: {text!r}") from exc
