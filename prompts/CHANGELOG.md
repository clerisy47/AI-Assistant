# Prompt changelog (verified research)

Each version must respond to a **specific failure** seen in prior harness / MLflow traces.

## prompt_v1

**Baseline.** Extracted from the embedded research system prompt used through Phase 10.
No prior experiment diagnosis — starting point for MLflow comparison.

## prompt_v2

**Failure seen (v1 traces):** Case `weak_answer_soft` — stop_reason `verified` but soft failure
because the draft said only that local models are supported and omitted the
**OpenAI-compatible** protocol name required by the rubric / corpus claims.

**Change:** Added explicit draft-quality rules: name protocols/providers from notes
(vLLM, OpenAI-compatible API) instead of vague paraphrases.

**Config delta (matrix):** `TOP_K=6` — richer first-hit retrieval when incomplete
notes previously forced a re-search path (`needs_re_search` traces).

## prompt_v3

**Failure seen (v2 traces / taxonomy):** Case `empty_retrieval_cascade` documents
cascading soft failure (empty retrieval → confident fabricated answer). Case
`clarification` shows ambiguous queries should stop early. Mean tokens rise with
deeper retrieval (`TOP_K=6` in v2).

**Change:** Prefer clarify-before-search on ambiguous questions; never invent
corpus facts after empty/failed retrieval; push faster note externalization.

**Config delta (matrix):** `TOOL_RESULT_MAX_CHARS=1500` (down from 2000) to curb
context/token growth after v2’s deeper retrieval, trading some raw chunk text
for leaner trajectories.
