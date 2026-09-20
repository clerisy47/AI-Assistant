# Verified research — eval report

Generated: 2026-09-20 02:52:19 UTC

## Summary

- Cases: **11**
- Task completion rate: **91%**
- Tool-call correctness: **100%**
- Mean iterations: **1.27**
- Mean tokens: **160.5**
- Failures: hard=0, soft=1, cascading_soft=1, none=9

## Per-case results

| case id | completed | tools ok | iterations | tokens | failure class | notes |
|---|---|---|---:|---:|---|---|
| sufficient_first_search | yes | yes | 1 | 160 | none |  |
| needs_re_search | yes | yes | 2 | 320 | none |  |
| clarification | yes | yes | 1 | 35 | none |  |
| verify_fail_then_revise | yes | yes | 2 | 320 | none |  |
| max_iterations_stop | yes | yes | 2 | 180 | none |  |
| max_tool_calls_stop | yes | yes | 1 | 70 | none |  |
| tool_order_search_then_notes | yes | yes | 1 | 160 | none |  |
| weak_answer_soft | no | yes | 1 | 160 | soft | taxonomy demo: soft (auto=soft) |
| baseline_token_compare | yes | yes | 1 | 160 (multi=160, single=110) | none |  |
| empty_retrieval_cascade | yes | yes | 1 | 125 | cascading_soft | taxonomy demo: cascading_soft (auto=cascading_soft) |
| kb_unavailable_recognized | yes | yes | 1 | 75 | none |  |

## Failure log

- **weak_answer_soft** (`soft`): stop=verified; taxonomy demo: soft (auto=soft)
- **empty_retrieval_cascade** (`cascading_soft`): stop=verified; taxonomy demo: cascading_soft (auto=cascading_soft)

## Baseline token comparison

- Case `baseline_token_compare`: multi-agent tokens=160, single-agent tokens=110.
- Coordination overhead (multi − single): **50** tokens (positive means multi-agent used more).

## Failure injection

Phase 7 demo (case `kb_unavailable_recognized`):

- **Injected:** `INJECT_FAILURE=kb_unavailable` (harness `inject_failure` on `search_knowledge_base`) — tool returns a structured error with `is_error=true`.
- **Observed:** agent acknowledges the outage (`unavailable` / `retry` in the answer), supervisor stops with `stop_reason=tool_failure`, and no corpus facts are invented.
- **Contrast:** case `empty_retrieval_cascade` scripts the opposite (empty retrieval → confident fabricated answer → verifier rubber-stamp) and is scored as `cascading_soft` to document hallucinate-vs-recognize.
