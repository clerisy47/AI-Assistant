"""Document chunking.

Implements sentence-aware "recursive" chunking: pack whole sentences into a
chunk up to `chunk_size` characters, then start the next chunk with a bit of
overlap carried over from the tail of the previous one, so a fact that lands
right on a chunk boundary isn't invisible to retrieval.

Character-based sizing (rather than a tokenizer) is a deliberate simplicity
trade-off -- it needs zero extra dependencies and is predictable to reason
about. If you need token-exact budgets for a specific embedding model's
context window, swap `len(text)` below for a `tiktoken`/model tokenizer
count; nothing else in the pipeline needs to change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def split_into_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping, sentence-aware chunks.

    Guarantees (exercised in tests/test_chunking.py):
      - No chunk exceeds `chunk_size` characters, even if a single sentence does
        (an oversized sentence is hard-split as a last resort).
      - Consecutive chunks overlap by roughly `chunk_overlap` characters.
      - Empty/whitespace-only input yields no chunks.
    """
    metadata = metadata or {}
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = split_into_sentences(text)
    chunks: list[Chunk] = []
    current = ""

    def flush(buffer: str) -> None:
        if buffer.strip():
            chunks.append(Chunk(text=buffer.strip(), index=len(chunks), metadata=dict(metadata)))

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Candidate would overflow: flush what we have, start a fresh chunk
        # carrying a bit of overlap from the tail of the previous one.
        flush(current)
        overlap_text = current[-chunk_overlap:].strip() if chunk_overlap and current else ""
        current = f"{overlap_text} {sentence}".strip() if overlap_text else sentence

        # A single sentence longer than chunk_size on its own: hard-split it.
        while len(current) > chunk_size:
            flush(current[:chunk_size])
            current = current[chunk_size - chunk_overlap :]

    flush(current)
    return chunks
