"""FastAPI dependency accessors.

Shared, expensive-to-construct objects (the embedding model, vector store,
retriever, ingestion service) are built once in `app.main`'s lifespan
handler and stashed on `app.state`. These functions just hand them to route
handlers via FastAPI's `Depends`, which also makes routes trivially
testable by overriding the dependency with a fake.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request

from app.rag.ingestion import IngestionService
from app.rag.retriever import Retriever


def get_retriever(request: Request) -> Optional[Retriever]:
    """May be None if the RAG stack failed to initialize at startup (see
    app/main.py's lifespan). Routes that need it MUST also depend on
    `vector_store_ready` and check it before calling anything on this."""
    return request.app.state.retriever


def get_ingestion(request: Request) -> Optional[IngestionService]:
    """See get_retriever -- same caveat, same required `vector_store_ready` guard."""
    return request.app.state.ingestion


def vector_store_ready(request: Request) -> bool:
    return bool(getattr(request.app.state, "vector_store_ready", False))
