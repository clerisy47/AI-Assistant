# RAG Pipeline — Ingestion and Chunking

This document describes how documents become searchable vectors in this
assistant. Defaults match `app/config.py` unless overridden in `.env`.

## Ingestion flow

1. **Read** — supported formats include `.txt`, `.md`, and `.pdf`.
2. **Chunk** — text is split into overlapping windows so related sentences
   stay together across boundaries.
3. **Embed** — each chunk is embedded locally with the sentence-transformers
   model `BAAI/bge-small-en-v1.5` (setting `EMBEDDING_MODEL`).
4. **Upsert** — vectors are stored in **Qdrant** (`QDRANT_URL`, collection
   `QDRANT_COLLECTION`, default name `knowledge_base`).

Bulk ingest of the bundled corpus:

```bash
make ingest
# or: docker compose exec app python scripts/ingest_sample_docs.py sample_docs
```

## Chunking defaults

| Setting | Default | Role |
|---|---|---|
| `CHUNK_SIZE` | **800** | Maximum characters per chunk |
| `CHUNK_OVERLAP` | **120** | Characters shared with the previous chunk |

Overlap reduces the chance that a claim straddling a boundary is lost.
Retrieval returns the top-`TOP_K` (default **4**) nearest chunks for a query.

## Local LLM note (consistency with other docs)

The same OpenAI-compatible client used for cloud OpenAI also talks to a local
**vLLM** server when `LLM_PROVIDER=local`. Local serving does not change the
RAG path: embeddings and Qdrant still run locally; only the chat/completion
provider base URL changes.
