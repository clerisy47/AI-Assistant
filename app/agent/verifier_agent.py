"""Verifier agent — isolated claim check against EvidenceNotes (Phase 4).

Receives question + draft + structured notes only (no exploratory tool dump).
Uses generate_structured for a flat VerificationResult; no verifier tool loop.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.evidence_notes import EvidenceNotes, to_verifier_context
from app.llm.base import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """\
You are a strict claim verifier for corpus-backed research answers.

Given a user question, a draft answer, and structured evidence notes, decide whether \
every material claim in the draft is supported by the notes.

Rules:
- Prefer insufficient when notes are empty, gaps remain, or claims lack excerpts.
- Do not invent corpus facts that are absent from the notes.
- suggested_next_action must be one of: finalize, re_research, clarify.
- If sufficient is true, suggested_next_action must be finalize.
- If sufficient is false, prefer re_research unless the question itself is ambiguous (clarify).
"""


class VerificationResult(BaseModel):
    """Flat schema for provider structured output (no nested models / $defs)."""

    sufficient: bool = Field(description="True only if the draft is fully supported by the notes.")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the draft that lack support in the evidence notes.",
    )
    suggested_next_action: Literal["finalize", "re_research", "clarify"] = Field(
        description="Next supervisor action.",
    )
    notes: str = Field(default="", description="Short rationale for the verification decision.")


class VerifierAgent:
    def __init__(self, provider: LLMProvider):
        self._provider = provider

    @staticmethod
    def build_user_content(question: str, draft_answer: str, notes: EvidenceNotes) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Draft answer:\n{draft_answer or '(empty)'}\n\n"
            f"Evidence notes:\n{to_verifier_context(notes)}\n\n"
            "Return a structured verification decision."
        )

    def _normalize(self, result: VerificationResult) -> VerificationResult:
        if result.sufficient:
            return result.model_copy(update={"suggested_next_action": "finalize"})
        if result.suggested_next_action == "finalize":
            return result.model_copy(update={"suggested_next_action": "re_research"})
        return result

    async def verify(
        self,
        *,
        question: str,
        draft_answer: str,
        evidence_notes: EvidenceNotes,
        temperature: float = 0.0,
    ) -> tuple[VerificationResult, dict]:
        """Return (result, tool_trace_entry). Context is draft + notes only."""
        user_content = self.build_user_content(question, draft_answer, evidence_notes)
        messages = [LLMMessage(role="user", content=user_content)]
        schema = VerificationResult.model_json_schema()

        try:
            data = await self._provider.generate_structured(
                messages,
                schema=schema,
                schema_name="verification_result",
                system=VERIFIER_SYSTEM_PROMPT,
                temperature=temperature,
            )
            result = self._normalize(VerificationResult.model_validate(data))
            is_error = False
            result_text = result.model_dump_json()
        except Exception as exc:  # noqa: BLE001 — fail closed into re-research
            logger.warning("Verifier structured output failed: %s", exc)
            result = VerificationResult(
                sufficient=False,
                unsupported_claims=["verification_unavailable"],
                suggested_next_action="re_research",
                notes=f"Verifier failed: {exc}",
            )
            is_error = True
            result_text = json.dumps({"error": str(exc), "fallback": result.model_dump()})

        trace_entry = {
            "agent": "verifier",
            "tool": "verify_claims",
            "arguments": {
                "question": question,
                "draft_chars": len(draft_answer or ""),
                "evidence_items": len(evidence_notes.items),
            },
            "result": result_text,
            "is_error": is_error,
        }
        return result, trace_entry
