"""Exposes the vector-store knowledge base to the model *as a tool*.

This is a deliberate design choice: rather than always prepending retrieved
chunks to every prompt ("naive RAG"), the `/chat` agent decides for itself,
per turn, whether a question needs the knowledge base at all -- a greeting
or a math question shouldn't burn a retrieval call. The classic
retrieve-then-generate pattern (retrieve unconditionally, then answer) is
still available at `/rag/query` for comparison; see app/api/rag.py.
"""

from __future__ import annotations

from typing import Optional

from app.llm.base import ToolDefinition
from app.rag.retriever import Retriever

KNOWLEDGE_BASE_DEFINITION = ToolDefinition(
    name="search_knowledge_base",
    description=(
        "Search the ingested document knowledge base for information relevant to a query. "
        "Use this whenever the user asks something that might be answered by ingested documents "
        "-- do not rely on general knowledge for questions about the user's own documents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "top_k": {"type": "integer", "description": "How many chunks to retrieve. Use 4 if unsure."},
        },
        "required": ["query", "top_k"],
        "additionalProperties": False,
    },
)


def make_knowledge_base_tool(retriever: Optional[Retriever]):
    """Bind a `search_knowledge_base` handler to a specific Retriever instance
    (closures give us cheap dependency injection without a DI framework).

    `retriever` may be None if the RAG stack failed to initialize at startup
    (see app/main.py) -- handled here, rather than left to raise, so the
    agent loop gets a clear tool result it can react to (e.g. apologize and
    answer from general knowledge) instead of an opaque AttributeError.
    """

    async def search_knowledge_base(query: str, top_k: int = 4) -> str:
        if retriever is None:
            return (
                "The knowledge base is currently unavailable (vector store did not initialize). "
                "Tell the user document search isn't working right now instead of guessing at an answer."
            )
        results = await retriever.retrieve(query, top_k=top_k or 4)
        if not results:
            return "No relevant documents found in the knowledge base."
        return "\n\n".join(
            f"[source: {r.metadata.get('source', 'unknown')} | relevance={r.score:.3f}]\n{r.text}" for r in results
        )

    return search_knowledge_base
