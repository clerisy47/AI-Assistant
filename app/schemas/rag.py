from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.config import settings


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.TOP_K, ge=1, le=20)


class SourceChunk(BaseModel):
    source: Optional[str]
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
