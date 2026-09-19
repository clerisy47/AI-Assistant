"""The agent loop: the actual mechanics of "tool calling" as a conversation.

One call to `AgentOrchestrator.run()`:
  1. Sends the conversation + available tools to the LLM.
  2. If the model replies with plain text, we're done.
  3. If it replies with one or more tool calls, execute each one, append the
     assistant's tool-call message *and* the tool results to the
     conversation, and go back to step 1.
  4. Stops after `max_tool_iterations` round-trips as a safety valve against
     an infinite tool-call loop (e.g. a model repeatedly calling a tool with
     the same bad arguments).

Every tool call and its result is collected into `tool_trace` and returned
alongside the final answer, so a caller (or a developer debugging a bad
answer) can see exactly what the model looked up and what it got back --
this is the difference between "the model said X" and "the model said X
because tool Y returned Z".
"""

from __future__ import annotations

import logging

from app.llm.base import LLMMessage, LLMProvider
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are a precise, honest AI assistant with access to tools.

Guidelines:
- Use `search_knowledge_base` whenever the user's question could be answered by ingested \
documents, rather than relying on general knowledge. Cite the source file(s) it returns \
when you use them.
- Use `calculator` for any arithmetic rather than computing it yourself.
- Use `get_current_datetime` for any question about the current date, time, or "today".
- If a tool returns no useful result, say so plainly instead of guessing or making \
something up.
- Keep answers concise and directly responsive to the question asked.
"""


class AgentOrchestrator:
    def __init__(self, provider: LLMProvider, tools: ToolRegistry, max_tool_iterations: int = 5):
        self._provider = provider
        self._tools = tools
        self._max_iterations = max_tool_iterations

    async def run(
        self,
        history: list[LLMMessage],
        *,
        system: str | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1500,
    ) -> dict:
        messages = list(history)
        trace: list[dict] = []
        tool_defs = self._tools.definitions()
        system_prompt = system or DEFAULT_SYSTEM_PROMPT

        for iteration in range(self._max_iterations):
            response = await self._provider.generate(
                messages,
                system=system_prompt,
                tools=tool_defs or None,
                tool_choice="auto",
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            if not response.tool_calls:
                return {"answer": response.content or "", "tool_trace": trace, "iterations": iteration + 1}

            messages.append(LLMMessage(role="assistant", content=response.content, tool_calls=response.tool_calls))

            for call in response.tool_calls:
                result_text, is_error = await self._tools.call(call.name, call.arguments)
                trace.append({"tool": call.name, "arguments": call.arguments, "result": result_text, "is_error": is_error})
                messages.append(
                    LLMMessage(role="tool", tool_call_id=call.id, name=call.name, content=result_text, is_error=is_error)
                )

        logger.warning("Agent hit max_tool_iterations (%d) without a final answer", self._max_iterations)
        return {
            "answer": "I wasn't able to reach a final answer within the allowed number of tool-use steps. "
            "Please try rephrasing your question.",
            "tool_trace": trace,
            "iterations": self._max_iterations,
        }
