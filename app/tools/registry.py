"""A small registry mapping tool name -> (JSON-schema definition, async handler).

The agent orchestrator only knows about this registry, not about individual
tools -- adding a new tool means writing one function and one `register()`
call, nothing else in the request-handling path changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.llm.base import ToolDefinition

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Awaitable[str]]


@dataclass
class RegisteredTool:
    definition: ToolDefinition
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self._tools[definition.name] = RegisteredTool(definition=definition, handler=handler)

    def definitions(self) -> list[ToolDefinition]:
        return [t.definition for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Execute a registered tool. Returns (result_text, is_error).

        Errors are caught and turned into a result string rather than raised,
        because the whole point is to feed the outcome -- success or failure
        -- back to the model as a tool_result so it can react (retry with
        different arguments, apologize, try another tool, etc.) instead of
        crashing the request.
        """
        if name not in self._tools:
            return f"Error: unknown tool '{name}'.", True
        try:
            result = await self._tools[name].handler(**arguments)
            return result, False
        except TypeError as exc:
            logger.warning("Tool '%s' called with bad arguments %r: %s", name, arguments, exc)
            return f"Error: invalid arguments for tool '{name}': {exc}", True
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: tool failures must not crash the request
            logger.exception("Tool '%s' raised an exception", name)
            return f"Error executing tool '{name}': {exc}", True
