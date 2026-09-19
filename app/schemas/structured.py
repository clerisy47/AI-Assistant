"""Schemas for the structured-output demo endpoint.

`DocumentSummary` is deliberately a flat model (no nested BaseModels) so its
`model_json_schema()` output needs no `$defs`/`$ref` resolution before being
handed to a provider's structured-output feature -- see
`app.llm.base.strict_json_schema` and the README's note on nested schemas.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to summarize.")


class DocumentSummary(BaseModel):
    title: str = Field(description="A short, descriptive title for the text.")
    key_points: list[str] = Field(description="3-6 bullet-point takeaways.")
    sentiment: Literal["positive", "neutral", "negative"]
    word_count_estimate: int = Field(description="Approximate word count of the original text.")
