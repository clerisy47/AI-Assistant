"""Health check.

Deliberately does *not* ping the LLM provider or Qdrant on every call --
that would make `/health` slow, costly (burns LLM tokens on every load
balancer probe), and would make it require live infrastructure to unit
test. It reports the resolved configuration and whether the vector store
initialized successfully at startup, which is enough to answer "is this
service sanely configured and up" without side effects.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    active_model = {
        "anthropic": settings.ANTHROPIC_MODEL,
        "openai": settings.OPENAI_MODEL,
        "local": settings.VLLM_MODEL,
    }.get(settings.LLM_PROVIDER.lower(), "unknown")

    return {
        "status": "ok",
        "llm_provider": settings.LLM_PROVIDER,
        "active_model": active_model,
        "vector_db_url": settings.QDRANT_URL,
        "vector_store_ready": bool(getattr(request.app.state, "vector_store_ready", False)),
        "embedding_model": settings.EMBEDDING_MODEL,
    }
