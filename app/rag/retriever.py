"""Query-time retrieval: embed a query, search the vector store, format results."""

from __future__ import annotations

from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import SearchResult, VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self._store = vector_store
        self._embedder = embedding_model

    async def retrieve(self, query: str, top_k: int = 4) -> list[SearchResult]:
        vectors = await self._embedder.embed([query])
        if not vectors:
            return []
        return await self._store.search(vectors[0], top_k=top_k)
