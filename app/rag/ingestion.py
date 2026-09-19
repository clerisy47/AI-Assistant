"""Document ingestion pipeline: read -> chunk -> embed -> upsert.

Supports .txt, .md, and .pdf out of the box. Adding a new format means
adding one branch to `read_file` -- chunking, embedding, and storage are
already format-agnostic since they only ever see plain text.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from pypdf import PdfReader

from app.rag.chunking import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


class IngestionService:
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel, chunk_size: int, chunk_overlap: int):
        self._store = vector_store
        self._embedder = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest_text(self, text: str, source: str) -> int:
        chunks = chunk_text(text, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap, metadata={"source": source})
        if not chunks:
            return 0
        vectors = await self._embedder.embed([c.text for c in chunks])
        ids = [str(uuid.uuid4()) for _ in chunks]
        payloads = [{"text": c.text, "source": source, "chunk_index": c.index} for c in chunks]
        await self._store.upsert(ids, vectors, payloads)
        logger.info("Ingested %d chunks from '%s'", len(chunks), source)
        return len(chunks)

    async def ingest_file(self, path: Path, source: str | None = None) -> int:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
        text = read_file(path)
        return await self.ingest_text(text, source=source or path.name)

    async def ingest_directory(self, directory: Path) -> dict[str, int]:
        results: dict[str, int] = {}
        for path in sorted(Path(directory).glob("**/*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                results[path.name] = await self.ingest_file(path)
        return results
