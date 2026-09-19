#!/usr/bin/env python
"""Bulk-ingest every supported document in a directory into the vector store.

Usage (from inside the running app container):
    docker compose exec app python scripts/ingest_sample_docs.py sample_docs

Usage (running locally, against a Qdrant reachable at QDRANT_URL):
    python scripts/ingest_sample_docs.py sample_docs
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.rag.embeddings import EmbeddingModel  # noqa: E402
from app.rag.ingestion import IngestionService  # noqa: E402
from app.rag.vector_store import QdrantVectorStore  # noqa: E402


async def main() -> None:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_docs")
    if not directory.is_dir():
        print(f"'{directory}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    embedder = await EmbeddingModel.get(settings.EMBEDDING_MODEL)
    store = QdrantVectorStore(url=settings.QDRANT_URL, collection=settings.QDRANT_COLLECTION)
    await store.ensure_collection(embedder.dimension)

    ingestion = IngestionService(store, embedder, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    results = await ingestion.ingest_directory(directory)

    if not results:
        print(f"No supported documents (.txt, .md, .pdf) found in '{directory}'.")
        return

    for name, count in results.items():
        print(f"  {name}: {count} chunks")
    print(f"\nIngested {sum(results.values())} chunks from {len(results)} file(s) into collection '{settings.QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    asyncio.run(main())
