# MLflow experiment comparison (Phase 11)

Generated: 2026-09-20 03:17:49 UTC

Scripted harness runs over [`eval/cases.yaml`](../../eval/cases.yaml) with versioned prompts under [`prompts/`](../../prompts/) and config overrides from [`experiment_matrix.yaml`](../experiment_matrix.yaml).

## Results

| prompt | config overrides | completion | tool ok | mean iter | mean tokens | soft fails | run_id |
|---|---|---:|---:|---:|---:|---:|---|
| `prompt_v1` | (defaults) | 91% | 100% | 1.27 | 160.5 | 1 | `6d9085eafe764b1897bfd6c42348b514` |
| `prompt_v2` | TOP_K=6 | 91% | 100% | 1.27 | 160.5 | 1 | `d1d3c117c5ae458789143588fd18d8a8` |
| `prompt_v3` | TOP_K=6, TOOL_RESULT_MAX_CHARS=1500 | 91% | 100% | 1.27 | 160.5 | 1 | `4d45b73b13e2421f881f0239690543a2` |

## Winner and trade-off

**Winner:** `prompt_v3` (completion=91%, mean_tokens=160.5). v3 keeps v2’s retrieval depth and draft-naming rules while tightening `TOOL_RESULT_MAX_CHARS` for leaner context; trade-off is less raw chunk text in the research agent window versus v1 defaults.

## Notes

- Traces are custom JSONL artifacts (not MLflow autolog).
- Prompt text changes do not alter scripted provider queues; config params still differ per run for comparable MLflow params/metrics.
- Diagnoses that drove v2/v3: see [`prompts/CHANGELOG.md`](../../prompts/CHANGELOG.md).
- Reload locally: `MLFLOW_TRACKING_URI=./mlruns mlflow ui` after `make mlflow-experiment`.
