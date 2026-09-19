"""Application entrypoint.

Startup (via the `lifespan` context manager -- the current FastAPI pattern,
replacing the deprecated `@app.on_event("startup")`) builds the embedding
model, vector store, retriever, and ingestion service exactly once and
attaches them to `app.state`.

Two independent things can go wrong here -- the embedding-model load (first
run: downloads weights over the network) and the Qdrant connection (not
accepting connections the instant this container starts) -- and neither is
allowed to crash the whole app or hang it indefinitely:

- Failure (of either) is caught, logged, and turned into
  `vector_store_ready = False` rather than a crash. `/chat` still works
  through tools that don't need retrieval (calculator, datetime); `/rag/*`
  endpoints return a clear 503 instead of a crash.
- The whole attempt is bounded by `RAG_STARTUP_TIMEOUT_SECONDS`, and the
  slow embedding-model construction runs in a worker thread (see
  `EmbeddingModel.get`) specifically so that bound can actually be enforced
  -- a timeout wrapped around a *synchronous* blocking call can't fire,
  because the event loop is frozen for the call's entire duration and never
  gets a chance to notice the clock ran out.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, health, rag, research, structured
from app.config import settings
from app.rag.embeddings import EmbeddingModel
from app.rag.ingestion import IngestionService
from app.rag.retriever import Retriever
from app.rag.vector_store import QdrantVectorStore

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Generous enough for a real first-run model download over a slow-but-working
# connection and for Qdrant's own internal retry/backoff (up to ~16s) to
# resolve, but still a hard ceiling so a fully unreachable network can't
# hang container startup indefinitely.
RAG_STARTUP_TIMEOUT_SECONDS = 45.0


async def _init_rag_stack() -> tuple[Retriever, IngestionService]:
    embedder = await EmbeddingModel.get(settings.EMBEDDING_MODEL)
    vector_store = QdrantVectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)
    await vector_store.ensure_collection(embedder.dimension)
    retriever = Retriever(vector_store, embedder)
    ingestion = IngestionService(vector_store, embedder, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    return retriever, ingestion


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_store_ready = False
    app.state.retriever = None
    app.state.ingestion = None

    try:
        retriever, ingestion = await asyncio.wait_for(_init_rag_stack(), timeout=RAG_STARTUP_TIMEOUT_SECONDS)
        app.state.retriever = retriever
        app.state.ingestion = ingestion
        app.state.vector_store_ready = True
    except asyncio.TimeoutError:
        logger.error(
            "RAG stack did not finish initializing within %.0fs (embedding model download and/or the "
            "Qdrant connection at %s). Continuing startup without it -- /rag/* endpoints will return 503 "
            "until this resolves. This is usually a first-run model download over a slow or unreachable "
            "network; once '%s' is cached, restarts will be fast.",
            RAG_STARTUP_TIMEOUT_SECONDS, settings.QDRANT_URL, settings.EMBEDDING_MODEL,
        )
    except Exception:
        logger.exception(
            "RAG stack failed to initialize (embedding model load and/or the Qdrant connection at %s). "
            "The server will still start. /rag/* endpoints and the search_knowledge_base tool will "
            "return a clear error until this resolves; other chat tools are unaffected.",
            settings.QDRANT_URL,
        )

    yield
    # No explicit teardown: the Qdrant/embedding clients hold no resources
    # that need an orderly close for this app's lifetime.


app = FastAPI(
    title="AI Assistant -- RAG + Tool Calling",
    description="A provider-agnostic LLM assistant with retrieval-augmented generation, tool calling, and structured output.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(research.router)
app.include_router(rag.router)
app.include_router(structured.router)
