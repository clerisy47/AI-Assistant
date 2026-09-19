"""Vector store abstraction, with Qdrant as the concrete implementation.

Behind a small `VectorStore` ABC so the embedding/retrieval code never talks
to the Qdrant client directly -- swapping in a different vector database
later means writing one new class, not touching the RAG pipeline.

Two things worth calling out:

- Uses `query_points` (Qdrant's current Query API), not the older `search`
  method, which the qdrant-client library has deprecated.
- `ensure_collection` retries with backoff. The official `qdrant/qdrant`
  Docker image ships without curl/wget/a shell utility suitable for a
  Compose `healthcheck:` (this is a known, deliberate choice upstream --
  see qdrant/qdrant#4250), so `depends_on` can only wait for the container
  to *start*, not for Qdrant to actually be ready to accept connections.
  Retrying here, at the one place the app first talks to Qdrant, is what
  actually makes startup ordering robust.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    async def ensure_collection(self, dimension: int) -> None: ...

    @abstractmethod
    async def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    async def search(self, vector: list[float], top_k: int = 4) -> list[SearchResult]: ...


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str, collection: str):
        self._client = AsyncQdrantClient(url=url)
        self._collection = collection

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_fixed(2),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def ensure_collection(self, dimension: int) -> None:
        existing = [c.name for c in (await self._client.get_collections()).collections]
        if self._collection not in existing:
            logger.info("Creating Qdrant collection '%s' (dim=%d)", self._collection, dimension)
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    async def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)]
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def search(self, vector: list[float], top_k: int = 4) -> list[SearchResult]:
        response = await self._client.query_points(collection_name=self._collection, query=vector, limit=top_k)
        return [
            SearchResult(text=point.payload.get("text", ""), score=point.score, metadata=point.payload or {})
            for point in response.points
        ]
