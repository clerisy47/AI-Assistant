---
name: mlops-tracking
description: >-
  Implements Track A MLOps for this assistant: uv (pyproject.toml + uv.lock),
  MLflow prompt/config experiment runs with full agent step traces, Evidently
  LLM-as-judge regression against a golden set, and optional Airflow scheduled
  eval. Use when migrating to uv, logging MLflow runs, building Evidently
  suites, or wiring the regression DAG (docs/SPECS.md Phases 10–14).
---

# MLOps Tracking (Track A)

No trained model — track **prompts, retrieval/agent config, and behavior**.

## Before coding

Read `docs/SPECS.md` Part II (Phases 10–14) and root `AGENTS.md` items 7–8.

## Phase cheatsheet

| Phase | Do |
|---|---|
| 10 | `pyproject.toml` + `uv.lock`; Makefile/Docker/README → `uv sync` / `uv run` |
| 11 | `prompts/prompt_v{1,2,3}.md` + CHANGELOG; step traces; MLflow params/metrics/artifacts; ≥3 runs |
| 12 | `eval/golden_set.yaml`; Evidently Test Suite (≥2 checks incl. reference correctness); HTML reports; log `pct_tests_passed` |
| 13 | Airflow DAG + degradation threshold; ship dry-run if full Airflow is heavy |
| 14 | README a–d (uv / MLflow / Evidently / orchestration) + architecture + `mlops/reports/` |

## Hard rules

1. Each `prompt_vN` must fix a **named failure** from prior traces (`prompts/CHANGELOG.md`).
2. Traces are **step-level** (`tool`, `args`, `result`, `reasoning`, `stop_reason`) — not final answer only.
3. Per version, save ≥2–3 trace artifacts (include a **failure** case).
4. Evidently failures are **regressions**; do not promote below `EVIDENTLY_PASS_THRESHOLD`.
5. Log Evidently pass rate onto the **same** MLflow run as harness metrics.

## Layout

```text
prompts/
mlops/
  tracking.py
  evidently_regression.py
  airflow/dags/regression_eval_dag.py
  reports/
eval/golden_set.yaml
pyproject.toml
uv.lock
```

## Done when

- Clean clone: `uv sync && uv run pytest`
- MLflow shows ≥3 comparable runs with traces
- Evidently HTML exists; `pct_tests_passed` in MLflow
- README a–d describe *this* system’s choices and trade-offs
