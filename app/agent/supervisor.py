"""Supervisor coordinating Research ↔ Verifier until stop (Phase 4+).

Owns global iteration / tool-call budgets and token accounting (Phase 5).
Verifier sees draft + EvidenceNotes only. Never wraps classic /rag/query
in a fixed pipeline. Optional baseline=single runs research tools without
a separate verifier for multi- vs single-agent token comparison.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from app.agent.evidence_notes import EvidenceNotes
from app.agent.research_agent import ResearchAgent
from app.agent.verifier_agent import VerificationResult, VerifierAgent
from app.config import settings
from app.llm.base import LLMProvider
from app.llm.usage import TokenUsage
from app.rag.retriever import Retriever
from app.tools.research_tools import ClarificationState, build_research_registry

logger = logging.getLogger(__name__)

BaselineMode = Literal["multi", "single"]

_PARTIAL_MAX_ITERATIONS = (
    "I wasn't able to finish verified research within the allowed number of steps. "
    "Here is a partial answer based on evidence gathered so far."
)
_PARTIAL_MAX_TOOL_CALLS = (
    "I hit the tool-call budget before finishing verified research. "
    "Here is a partial answer based on evidence gathered so far."
)

_RESEARCH_STOP_TO_API = {
    "final": "verified",
    "clarification": "clarification",
    "max_iterations": "max_iterations",
    "max_tool_calls": "max_tool_calls",
}


class ResearchSupervisor:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        retriever: Optional[Retriever] = None,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
        max_pass_iterations: int | None = None,
    ):
        self._provider = provider
        self._retriever = retriever
        self._max_iterations = (
            settings.MAX_RESEARCH_ITERATIONS if max_iterations is None else max_iterations
        )
        self._max_tool_calls = (
            settings.MAX_RESEARCH_TOOL_CALLS if max_tool_calls is None else max_tool_calls
        )
        self._max_pass_iterations = (
            settings.MAX_RESEARCH_PASS_ITERATIONS
            if max_pass_iterations is None
            else max_pass_iterations
        )
        self._verifier = VerifierAgent(provider)

    @classmethod
    def create(
        cls,
        provider: LLMProvider,
        *,
        retriever: Optional[Retriever] = None,
        max_iterations: int | None = None,
        max_tool_calls: int | None = None,
        max_pass_iterations: int | None = None,
    ) -> ResearchSupervisor:
        return cls(
            provider,
            retriever=retriever,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_pass_iterations=max_pass_iterations,
        )

    def _response(
        self,
        *,
        answer: str,
        verification: VerificationResult | None,
        evidence_notes: EvidenceNotes,
        tool_trace: list[dict],
        iterations: int,
        stop_reason: str,
        token_usage: TokenUsage,
    ) -> dict:
        return {
            "answer": answer,
            "verification": verification.model_dump() if verification else None,
            "evidence_notes": evidence_notes.model_dump(),
            "tool_trace": tool_trace,
            "iterations": iterations,
            "stop_reason": stop_reason,
            "token_usage": token_usage.to_dict(),
        }

    def _partial_answer(self, prefix: str, draft: str, notes: EvidenceNotes) -> str:
        if draft and draft.strip():
            return f"{prefix}\n\n{draft.strip()}"
        if not notes.items:
            return prefix
        lines = [prefix, "", "Evidence so far:"]
        for item in notes.items:
            lines.append(f"- [{item.source}] {item.excerpt}")
        return "\n".join(lines)

    def _revise_hints(self, verification: VerificationResult) -> list[str]:
        hints = [c for c in verification.unsupported_claims if (c or "").strip()]
        if verification.notes and verification.notes.strip():
            hints.append(verification.notes.strip())
        if not hints:
            hints.append("Evidence was judged insufficient; search again and revise the draft.")
        return hints

    def _as_agent_usage(self, usage: TokenUsage, agent: str) -> TokenUsage:
        """Attach totals under by_agent without double-counting prompt/completion."""
        return TokenUsage().add(usage, agent=agent)

    async def _run_single(
        self,
        question: str,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int | None,
    ) -> dict:
        """Single-agent baseline: research tools only, no separate verifier."""
        notes = EvidenceNotes(question=question)
        clarification = ClarificationState()
        registry, clarification = build_research_registry(
            self._retriever, notes, clarification
        )
        research = ResearchAgent(
            self._provider,
            registry,
            evidence_notes=notes,
            clarification=clarification,
            max_iterations=self._max_iterations,
            max_tool_calls=self._max_tool_calls,
        )
        research_result = await research.run(
            question,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        research_usage: TokenUsage = research_result["token_usage"]
        totals = self._as_agent_usage(research_usage, "research")
        tool_trace = list(research_result["tool_trace"])
        for entry in tool_trace:
            entry.setdefault("agent", "research")
        stop = _RESEARCH_STOP_TO_API.get(
            research_result["stop_reason"], research_result["stop_reason"]
        )
        return self._response(
            answer=research_result.get("draft_answer") or "",
            verification=None,
            evidence_notes=notes,
            tool_trace=tool_trace,
            iterations=research_result["iterations"],
            stop_reason=stop,
            token_usage=totals,
        )

    async def run(
        self,
        question: str,
        *,
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        baseline: BaselineMode = "multi",
    ) -> dict:
        if baseline == "single":
            return await self._run_single(
                question,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        notes = EvidenceNotes(question=question)
        tool_trace: list[dict] = []
        tool_calls_used = 0
        last_draft = ""
        last_verification: VerificationResult | None = None
        revise_hints: list[str] | None = None
        supervisor_steps = 0
        totals = TokenUsage()

        for _ in range(self._max_iterations):
            remaining_tools = self._max_tool_calls - tool_calls_used
            if remaining_tools <= 0:
                return self._response(
                    answer=self._partial_answer(_PARTIAL_MAX_TOOL_CALLS, last_draft, notes),
                    verification=last_verification,
                    evidence_notes=notes,
                    tool_trace=tool_trace,
                    iterations=supervisor_steps,
                    stop_reason="max_tool_calls",
                    token_usage=totals,
                )

            clarification = ClarificationState()
            registry, clarification = build_research_registry(
                self._retriever, notes, clarification
            )
            research = ResearchAgent(
                self._provider,
                registry,
                evidence_notes=notes,
                clarification=clarification,
                max_iterations=self._max_pass_iterations,
                max_tool_calls=remaining_tools,
            )

            research_result = await research.run(
                question,
                revise_hints=revise_hints,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            supervisor_steps += 1

            research_usage: TokenUsage = research_result["token_usage"]
            totals = totals.add(research_usage, agent="research")

            for entry in research_result["tool_trace"]:
                entry.setdefault("agent", "research")
                tool_trace.append(entry)
            tool_calls_used += len(research_result["tool_trace"])
            last_draft = research_result.get("draft_answer") or ""

            if research_result["stop_reason"] == "clarification":
                return self._response(
                    answer=last_draft,
                    verification=last_verification,
                    evidence_notes=notes,
                    tool_trace=tool_trace,
                    iterations=supervisor_steps,
                    stop_reason="clarification",
                    token_usage=totals,
                )

            if research_result["stop_reason"] == "max_tool_calls":
                return self._response(
                    answer=self._partial_answer(_PARTIAL_MAX_TOOL_CALLS, last_draft, notes),
                    verification=last_verification,
                    evidence_notes=notes,
                    tool_trace=tool_trace,
                    iterations=supervisor_steps,
                    stop_reason="max_tool_calls",
                    token_usage=totals,
                )

            verification, v_trace, v_usage = await self._verifier.verify(
                question=question,
                draft_answer=last_draft,
                evidence_notes=notes,
            )
            tool_trace.append(v_trace)
            totals = totals.add(v_usage, agent="verifier")
            last_verification = verification

            if verification.sufficient:
                return self._response(
                    answer=last_draft,
                    verification=verification,
                    evidence_notes=notes,
                    tool_trace=tool_trace,
                    iterations=supervisor_steps,
                    stop_reason="verified",
                    token_usage=totals,
                )

            if verification.suggested_next_action == "clarify":
                clarify_text = verification.notes.strip() or (
                    "Could you clarify the research question?"
                )
                return self._response(
                    answer=clarify_text,
                    verification=verification,
                    evidence_notes=notes,
                    tool_trace=tool_trace,
                    iterations=supervisor_steps,
                    stop_reason="clarification",
                    token_usage=totals,
                )

            # Default: re-research with verifier hints (same EvidenceNotes instance).
            revise_hints = self._revise_hints(verification)
            logger.info(
                "Supervisor step %d: verification insufficient; re-researching (%d hints)",
                supervisor_steps,
                len(revise_hints),
            )

        return self._response(
            answer=self._partial_answer(_PARTIAL_MAX_ITERATIONS, last_draft, notes),
            verification=last_verification,
            evidence_notes=notes,
            tool_trace=tool_trace,
            iterations=supervisor_steps or self._max_iterations,
            stop_reason="max_iterations",
            token_usage=totals,
        )
