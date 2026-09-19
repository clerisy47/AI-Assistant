---
name: eval-harness
description: >-
  Builds and runs the from-scratch evaluation harness for verified research
  (completion rate, tool-call correctness, trajectory length, tokens, failure
  taxonomy). Use when creating eval/cases, harness.py, reports, or failure injection.
---

# Evaluation Harness

Do **not** add LangSmith, RAGAS, DeepEval, or similar. Implement under `eval/`.

## Required metrics

| Metric | Record |
|---|---|
| Task completion rate | Success per case vs rubric / `stop_reason` |
| Tool-call correctness | Expected tools + valid args |
| Trajectory length | Iterations / tool calls; compare to `expected_max_iterations` |
| Token usage | Totals per query; multi-agent vs single-agent baseline when enabled |
| Failure log | `hard` / `soft` / `cascading_soft` |

## Layout

```text
eval/
  harness.py      # runner
  cases.yaml      # queries + expectations
  report.md       # latest Markdown table + failure log
```

## Failure taxonomy

- **Hard failure** — crash, empty final on budget with no honest partial, or fabricated success
- **Soft failure** — wrong tool once / weak answer / extra iteration; task still completes
- **Cascading soft failure** — early soft error causes later unsupported confident answer

## Failure injection

Support at least one mode (e.g. `INJECT_FAILURE=kb_unavailable`). Document whether the agent recognizes the failure vs hallucinates.

## Commands

Prefer wiring `make eval` → `python -m eval.harness` (or `python eval/harness.py`).

## Report shape

Markdown table: case id | completed | tools ok | iterations | tokens | failure class | notes.
