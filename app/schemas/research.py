"""Request/response schemas for POST /research (Phase 4)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.config import settings
from app.schemas.chat import ChatMessage


class ResearchRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    provider: Optional[Literal["anthropic", "openai", "local"]] = Field(
        default=None,
        description="Override the configured LLM_PROVIDER for this request only.",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    top_p: float = Field(default=settings.TOP_P, ge=0.0, le=1.0)
    max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        description="Override supervisor research↔verify round budget.",
    )


class VerificationPayload(BaseModel):
    sufficient: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    suggested_next_action: Optional[Literal["finalize", "re_research", "clarify"]] = None
    notes: str = ""


class ResearchToolTraceEntry(BaseModel):
    agent: Optional[Literal["research", "verifier"]] = None
    tool: str
    arguments: dict[str, Any]
    result: str
    is_error: bool


class ResearchResponse(BaseModel):
    answer: str
    verification: Optional[VerificationPayload] = None
    evidence_notes: dict[str, Any]
    tool_trace: list[ResearchToolTraceEntry]
    iterations: int
    stop_reason: Literal[
        "verified",
        "clarification",
        "max_iterations",
        "max_tool_calls",
        "tool_failure",
    ]
    token_usage: Optional[dict[str, Any]] = None
