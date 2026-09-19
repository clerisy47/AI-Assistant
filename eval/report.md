# Verified research — eval report

Generated: 2026-09-19 16:17:10 UTC

## Summary

- Cases: **10**
- Task completion rate: **90%**
- Tool-call correctness: **100%**
- Mean iterations: **1.30**
- Mean tokens: **169.0**
- Failures: hard=0, soft=1, cascading_soft=1, none=8

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

## Failure log

- **weak_answer_soft** (`soft`): stop=verified; taxonomy demo: soft (auto=soft)
- **empty_retrieval_cascade** (`cascading_soft`): stop=verified; taxonomy demo: cascading_soft (auto=cascading_soft)

## Baseline token comparison

- Case `baseline_token_compare`: multi-agent tokens=160, single-agent tokens=110.
- Coordination overhead (multi − single): **50** tokens (positive means multi-agent used more).
