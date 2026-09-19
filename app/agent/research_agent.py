"""Decision-capable research agent loop (Phase 3).

Iterates: search → evaluate sufficiency via evidence notes → search again /
clarify / draft. Hard-stopped by max_iterations and max_tool_calls.
Not wired to HTTP yet — Phase 4 supervisor will call ResearchAgent.run.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agent.context_budget import compact_messages
from app.agent.evidence_notes import EvidenceNotes
from app.config import settings
from app.llm.base import LLMMessage, LLMProvider
from app.rag.retriever import Retriever
from app.tools.registry import ToolRegistry
from app.tools.research_tools import ClarificationState, build_research_registry
from app.tools.skill_tool import build_skills_system_prefix

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """\
You are a corpus-backed research agent. Your job is to gather evidence from the \
knowledge base, record structured notes, and produce a grounded draft answer.

Guidelines:
- Call `load_skill` with "verified_research" before following the full research procedure.
- Use `search_knowledge_base` for document facts; do not invent corpus content.
- After useful retrieval, call `update_evidence_notes` with source, excerpt, and query_used.
- If notes leave open gaps, refine the query and search again (do not repeat the same query).
- If the user question is ambiguous, call `ask_clarification` and stop.
- When evidence is sufficient, reply with a plain-text draft answer (no tool call).
- If a tool errors or returns nothing useful, say so plainly instead of guessing.
"""

_PARTIAL_MAX_ITERATIONS = (
    "I wasn't able to finish researching within the allowed number of steps. "
    "Here is a partial answer based on evidence gathered so far."
)
_PARTIAL_MAX_TOOL_CALLS = (
    "I hit the tool-call budget before finishing research. "
    "Here is a partial answer based on evidence gathered so far."
)


class ResearchAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        evidence_notes: EvidenceNotes,
        clarification: ClarificationState,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
    ):
        self._provider = provider
        self._tools = tools
        self._notes = evidence_notes
        self._clarification = clarification
        self._max_iterations = (
            settings.MAX_RESEARCH_ITERATIONS if max_iterations is None else max_iterations
        )
        self._max_tool_calls = (
            settings.MAX_RESEARCH_TOOL_CALLS if max_tool_calls is None else max_tool_calls
        )

    @classmethod
    def create(
        cls,
        provider: LLMProvider,
        *,
        question: str,
        retriever: Optional[Retriever] = None,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
    ) -> ResearchAgent:
        """Convenience constructor with a fresh notes object and research registry."""
        notes = EvidenceNotes(question=question)
        registry, clarification = build_research_registry(retriever, notes)
        return cls(
            provider,
            registry,
            evidence_notes=notes,
            clarification=clarification,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
        )

    def _system_prompt(self) -> str:
        return f"{RESEARCH_SYSTEM_PROMPT}\n\n{build_skills_system_prefix()}"

    def _result(
        self,
        *,
        draft_answer: str,
        tool_trace: list[dict],
        iterations: int,
        stop_reason: str,
    ) -> dict:
        return {
            "draft_answer": draft_answer,
            "evidence_notes": self._notes,
            "tool_trace": tool_trace,
            "iterations": iterations,
            "stop_reason": stop_reason,
            "token_usage": None,
        }

    def _partial_from_notes(self, prefix: str) -> str:
        if not self._notes.items:
            return prefix
        lines = [prefix, "", "Evidence so far:"]
        for item in self._notes.items:
            lines.append(f"- [{item.source}] {item.excerpt}")
        return "\n".join(lines)

    async def run(
        self,
        question: str,
        *,
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> dict:
        if not self._notes.question:
            self._notes.question = question

        messages: list[LLMMessage] = [LLMMessage(role="user", content=question)]
        trace: list[dict] = []
        tool_defs = self._tools.definitions()
        system_prompt = self._system_prompt()
        tokens = max_tokens if max_tokens is not None else settings.MAX_TOKENS
        tool_calls_used = 0

        for iteration in range(self._max_iterations):
            response = await self._provider.generate(
                messages,
                system=system_prompt,
                tools=tool_defs or None,
                tool_choice="auto",
                temperature=temperature,
                top_p=top_p,
                max_tokens=tokens,
            )

            if not response.tool_calls:
                return self._result(
                    draft_answer=response.content or "",
                    tool_trace=trace,
                    iterations=iteration + 1,
                    stop_reason="final",
                )

            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            hit_tool_budget = False
            for call in response.tool_calls:
                if tool_calls_used >= self._max_tool_calls:
                    hit_tool_budget = True
                    break

                result_text, is_error = await self._tools.call(call.name, call.arguments)
                tool_calls_used += 1
                trace.append(
                    {
                        "tool": call.name,
                        "arguments": call.arguments,
                        "result": result_text,
                        "is_error": is_error,
                    }
                )
                messages.append(
                    LLMMessage(
                        role="tool",
                        tool_call_id=call.id,
                        name=call.name,
                        content=result_text,
                        is_error=is_error,
                    )
                )

                if call.name == "ask_clarification" and self._clarification.requested:
                    return self._result(
                        draft_answer=self._clarification.question or result_text,
                        tool_trace=trace,
                        iterations=iteration + 1,
                        stop_reason="clarification",
                    )

            if hit_tool_budget:
                logger.warning(
                    "Research agent hit max_tool_calls (%d)", self._max_tool_calls
                )
                return self._result(
                    draft_answer=self._partial_from_notes(_PARTIAL_MAX_TOOL_CALLS),
                    tool_trace=trace,
                    iterations=iteration + 1,
                    stop_reason="max_tool_calls",
                )

            # Cap context: once notes exist, compact older raw tool payloads.
            if self._notes.items:
                messages = compact_messages(messages, keep_last_n_tool_results=2)

        logger.warning(
            "Research agent hit max_iterations (%d) without a final answer",
            self._max_iterations,
        )
        return self._result(
            draft_answer=self._partial_from_notes(_PARTIAL_MAX_ITERATIONS),
            tool_trace=trace,
            iterations=self._max_iterations,
            stop_reason="max_iterations",
        )
