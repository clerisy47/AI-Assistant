"""Local embedding model.

Uses `sentence-transformers` with a small local model (default:
`BAAI/bge-small-en-v1.5`, 384-dim) rather than a cloud embeddings API. That
keeps the RAG pipeline runnable fully offline once the model weights are
cached -- no API key needed just to ingest and search documents -- which
pairs naturally with the local-vLLM deployment path.

`SentenceTransformer.encode()` is a blocking, CPU/GPU-bound call. Since the
rest of the app is async (FastAPI + async endpoints), embedding is offloaded
to a worker thread via `asyncio.to_thread` so it doesn't stall the event
loop while a batch is encoding.
"""

from __future__ import annotations

import asyncio
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    _instance: "EmbeddingModel | None" = None
    _lock = asyncio.Lock()

    def __init__(self, model_name: str):
        logger.info("Loading embedding model '%s' (first run downloads it; cache with a volume)", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @classmethod
    async def get(cls, model_name: str) -> "EmbeddingModel":
        """Process-wide singleton -- loading model weights is expensive
        enough that we want exactly one copy in memory, not one per request.

        Construction (and any first-run download) runs in a worker thread via
        `asyncio.to_thread`, not directly on the event loop. `SentenceTransformer(...)`
        is a slow, fully synchronous call -- run inline, it would freeze the
        event loop for as long as it takes, which also means a caller wrapping
        this in `asyncio.wait_for(...)` could never actually enforce a timeout
        (asyncio can only act at an `await` point; it can't interrupt a
        blocking call already in progress on the same thread). Running it in
        a separate thread is what makes that timeout actually work -- see
        `app/main.py`'s lifespan.
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = await asyncio.to_thread(cls, model_name)
        return cls._instance

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, texts)
