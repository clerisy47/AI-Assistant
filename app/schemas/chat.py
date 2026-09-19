from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.config import settings


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    provider: Optional[Literal["anthropic", "openai", "local"]] = Field(
        default=None, description="Override the configured LLM_PROVIDER for this request only."
    )
    temperature: float = Field(default=settings.TEMPERATURE, ge=0.0, le=1.0)
    top_p: float = Field(default=settings.TOP_P, ge=0.0, le=1.0)


class ToolTraceEntry(BaseModel):
    tool: str
    arguments: dict[str, Any]
    result: str
    is_error: bool


class ChatResponse(BaseModel):
    answer: str
    tool_trace: list[ToolTraceEntry]
    iterations: int
