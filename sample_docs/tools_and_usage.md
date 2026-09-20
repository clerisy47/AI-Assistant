# Tools and Usage Guidance

This document describes when the assistant should use each built-in tool.
The model chooses tools per turn; this file is corpus guidance for retrieval
and verified-research demos.

## Calculator

Use the **calculator** tool for arithmetic and numeric evaluation only —
expressions like sums, products, percentages, or unit conversions that do not
depend on project documentation. Do not use the calculator to answer questions
about architecture, configuration, or product behavior.

## Knowledge base search

Use **`search_knowledge_base`** when the answer should be grounded in ingested
documents (this corpus under `sample_docs/`, or any files the operator has
uploaded). Prefer it for:

- What the assistant supports (providers, RAG, local LLMs)
- How ingestion, chunking, or Qdrant is configured
- Comparing claims across documentation files

Do not invent corpus facts if retrieval returns nothing or errors; say so.

## Current datetime

Use **`get_current_datetime`** only when the user needs the wall-clock time or
date. It is not a substitute for searching the knowledge base or doing math.

## Choosing between calculator and knowledge-base

| Need | Tool |
|---|---|
| Numeric expression | calculator |
| Facts from project docs | search_knowledge_base |
| “What time is it?” | get_current_datetime |

If a question mixes docs and arithmetic (e.g. “chunk size is 800; what is
800 × 2?”), search first for the documented value, then use the calculator
for the expression.
