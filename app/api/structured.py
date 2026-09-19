"""Structured-output demo endpoint.

Turns free text into a `DocumentSummary` using each provider's native
constrained-decoding structured-output feature (see
`LLMProvider.generate_structured`), then validates the result through the
same Pydantic model on the way out -- belt and suspenders: the provider
guarantees schema-valid JSON, and Pydantic guarantees it also satisfies
Python-level types (e.g. the `Literal["positive","neutral","negative"]`
constraint on `sentiment`).

Temperature is pinned to 0: extraction/classification tasks like this
benefit from determinism, not creative variety.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider
from app.schemas.structured import DocumentSummary, SummarizeRequest

router = APIRouter(prefix="/structured", tags=["structured-output"])

STRUCTURED_TEMPERATURE = 0.0


@router.post("/summarize", response_model=DocumentSummary)
async def summarize(payload: SummarizeRequest) -> DocumentSummary:
    provider = get_llm_provider()
    schema = DocumentSummary.model_json_schema()
    messages = [LLMMessage(role="user", content=f"Summarize the following text:\n\n{payload.text}")]

    try:
        structured = await provider.generate_structured(
            messages, schema=schema, schema_name="document_summary", temperature=STRUCTURED_TEMPERATURE
        )
        return DocumentSummary.model_validate(structured.data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Failed to generate structured output: {exc}") from exc
