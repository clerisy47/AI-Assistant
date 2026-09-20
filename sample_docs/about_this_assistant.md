# About This Assistant

This is a sample document included so the retrieval-augmented generation (RAG)
pipeline can be exercised immediately after startup, without needing to supply
your own documents first.

## Architecture

The assistant is a FastAPI service backed by three main subsystems. The LLM
layer is provider-agnostic: it can call Anthropic Claude, OpenAI, or a locally
hosted open-source model served by vLLM, all through the same internal
interface. The RAG layer chunks ingested documents, embeds them with a local
sentence-transformers model, and stores the vectors in Qdrant. The agent layer
runs a multi-turn tool-calling loop so the model can decide, per question,
whether it needs to search the knowledge base, do arithmetic, or check the
current time.

## Vector database

This project uses Qdrant as its vector database. Qdrant runs as its own
container in the Docker Compose stack, listening on port 6333 for its REST
API. Embeddings are generated locally with the `BAAI/bge-small-en-v1.5`
sentence-transformers model, so the ingestion pipeline works fully offline
once the model weights are cached.

## Local model serving

For fully local operation, the project can serve an open-source model such as
Meta's Llama 3.1 8B Instruct with vLLM, using vLLM's OpenAI-compatible server.
Because the wire format is OpenAI-compatible, the same client code used for
the OpenAI cloud provider is reused for the local model -- only the base URL
changes. This is consistent with the RAG pipeline docs: local vLLM swaps the
LLM endpoint only; Qdrant and local embeddings stay as described under
vector database above.

## Try it

Ask the assistant something like "What vector database does this project
use?" or "How does local model serving work here?" -- a correct, grounded
answer means the retrieval pipeline found this file and used it.
