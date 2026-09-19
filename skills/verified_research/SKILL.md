---
name: verified_research
description: >-
  Procedure for corpus-backed research with iterative search, evidence notes,
  and handoff to verification. Load when answering comparison, consistency,
  or multi-source questions that may need more than one retrieval.
---

# Verified Research (runtime skill)

Short description for system prompt catalogs:
**Verified research** — iteratively search the knowledge base, record evidence notes, draft an answer, and request verification before finalizing.

## When relevant

- User asks to compare, confirm consistency, or research across documents.
- First retrieval looks thin, contradictory, or incomplete.
- Answer must be grounded in ingested corpus, not general knowledge.

## Procedure (full body — load only when selected)

1. **Clarify if needed.** If the question is underspecified (missing entity, ambiguous scope), call `ask_clarification` and stop.
2. **Search.** Call `search_knowledge_base` with a focused query. Prefer concrete nouns from the user question.
3. **Record notes.** Write `EvidenceNotes`: source file, short excerpt, query used. Cap excerpts; do not keep raw dumps.
4. **Evaluate sufficiency.** If notes leave open gaps, refine the query (different angle, filename hint, synonym) and search again. Do not repeat the identical query.
5. **Draft.** Produce a concise draft that only claims what notes support. Mark uncertainty explicitly.
6. **Handoff.** Signal ready for verification (supervisor routes to Verifier). Do not self-declare “verified.”
7. **On verify failure.** Address listed unsupported claims: search again or revise the draft. Respect global iteration budgets.

## Anti-patterns

- Single search then confident answer when notes are empty or off-topic.
- Copying entire retrieval payloads into the draft.
- Inventing facts when the tool errors or returns no hits.
