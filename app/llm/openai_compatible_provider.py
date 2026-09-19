"""OpenAI-compatible backend.

This single class serves *two* of the project's requirements at once:

1. "Connect to a major provider" -- instantiated with OpenAI's own
   base URL and an OpenAI API key, it talks to the real OpenAI API.
2. "Local deployment ... using vLLM" -- instantiated with
   `base_url="http://vllm:8000/v1"`, it talks to a self-hosted vLLM
   server instead, because vLLM's OpenAI-compatible server implements the
   same `/v1/chat/completions` wire format. No separate vLLM client is
   needed; only the base URL, API key, and model name change.

See app/llm/factory.py for how the two are constructed.

Structured output uses OpenAI's native `response_format={"type": "json_schema", ...}`
(constrained decoding, guaranteed-valid JSON) rather than a prompting trick.
vLLM implements the same `response_format` contract when started with a
guided-decoding backend (the default build supports this out of the box for
most models) -- see the README for caveats on the local path.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall, ToolDefinition, strict_json_schema

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        if not api_key and not base_url:
            logger.warning("OPENAI_API_KEY is not set; requests to OpenAI will fail.")
        # vLLM (and most local servers) ignore the API key entirely but the
        # SDK still requires a non-empty string to construct the client.
        self._client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)
        self._model = model

    # -- message translation -------------------------------------------------

    def _convert_messages(self, messages: list[LLMMessage], system: Optional[str]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        if system:
            converted.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "tool":
                converted.append(
                    {"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content or ""}
                )
            elif m.role == "assistant" and m.tool_calls:
                converted.append(
                    {
                        "role": "assistant",
                        "content": m.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                converted.append({"role": m.role, "content": m.content or ""})
        return converted

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": strict_json_schema(t.parameters) if t.strict else t.parameters,
                    "strict": t.strict,
                },
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
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "messages": self._convert_messages(messages, system),
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            if tool_choice in (None, "auto"):
                kwargs["tool_choice"] = "auto"
            elif tool_choice == "required":
                kwargs["tool_choice"] = "required"
            else:
                kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning("Model returned non-JSON tool arguments for %s: %r", tc.function.name, tc.function.arguments)
                arguments = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            usage=usage,
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
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=4096,
            messages=self._convert_messages(messages, system),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": strict_json_schema(schema), "strict": True},
            },
        )
        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON despite structured output mode: {content!r}") from exc
