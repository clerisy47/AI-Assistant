"""RAG endpoints.

Two distinct things live here on purpose:

- `POST /rag/ingest`: add a document to the knowledge base.
- `POST /rag/query`: classic single-shot RAG (retrieve, then generate one
  answer grounded in what was retrieved) -- useful on its own, and a good
  point of comparison against the agentic version of retrieval exposed as
  a *tool* in `/chat` (see app/tools/knowledge_base_tool.py).

Temperature is pinned low (0.2) here rather than using the general chat
default: for "answer using only this context" tasks, faithfulness to the
retrieved text matters more than variety in phrasing.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import get_ingestion, get_retriever, vector_store_ready
from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider
from app.rag.ingestion import SUPPORTED_EXTENSIONS, IngestionService
from app.rag.retriever import Retriever
from app.schemas.rag import IngestResponse, QueryRequest, QueryResponse, SourceChunk

router = APIRouter(prefix="/rag", tags=["rag"])

RAG_QUERY_TEMPERATURE = 0.2

QUERY_PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the answer is not contained in the context, say you don't know rather than guessing.

Context:
{context}

Question: {question}"""


def _require_vector_store(ready: bool) -> None:
    if not ready:
        raise HTTPException(status_code=503, detail="Vector store is not available. Check the Qdrant connection and /health.")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    ingestion: Optional[IngestionService] = Depends(get_ingestion),
    ready: bool = Depends(vector_store_ready),
) -> IngestResponse:
    _require_vector_store(ready)
    assert ingestion is not None  # guaranteed by the `ready` check above
    suffix = pathlib.Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        chunk_count = await ingestion.ingest_file(pathlib.Path(tmp.name), source=file.filename)

    return IngestResponse(filename=file.filename or "unknown", chunks_ingested=chunk_count)


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    retriever: Optional[Retriever] = Depends(get_retriever),
    ready: bool = Depends(vector_store_ready),
) -> QueryResponse:
    _require_vector_store(ready)
    assert retriever is not None  # guaranteed by the `ready` check above
    results = await retriever.retrieve(payload.query, top_k=payload.top_k)
    context = "\n\n".join(r.text for r in results) or "(no matching documents found)"

    provider = get_llm_provider()
    prompt = QUERY_PROMPT_TEMPLATE.format(context=context, question=payload.query)
    response = await provider.generate(
        [LLMMessage(role="user", content=prompt)],
        temperature=RAG_QUERY_TEMPERATURE,
        max_tokens=800,
    )

    return QueryResponse(
        answer=response.content or "",
        sources=[SourceChunk(source=r.metadata.get("source"), score=r.score, text=r.text[:300]) for r in results],
    )
