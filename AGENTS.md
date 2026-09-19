# AGENTS.md — AI Assistant

Read this before changing code. It orients human contributors and Cursor coding agents. The build plan lives in [`docs/SPECS.md`](docs/SPECS.md) — this file tells you what not to break and which skill/subagent to use.

## What this project is

A provider-agnostic **FastAPI LLM assistant** (Anthropic / OpenAI / local vLLM) with:

- Multi-turn tool calling (`POST /chat` → `AgentOrchestrator`)
- RAG over Qdrant (classic `/rag/query` and RAG-as-tool `search_knowledge_base`)
- Native structured output (`POST /structured/summarize`)

**Active extensions (see specs):**
- **Track B:** cross-source verified research — Research + Verifier under a supervisor
- **Track A (MLOps):** `uv` lockfile, MLflow prompt/config experiments with full step traces, Evidently LLM regression, optional Airflow scheduled eval

There is no trained model; MLOps tracks **prompts, retrieval/agent config, and behavior metrics**.

## Two meanings of “agent” (do not confuse them)

| Kind | Where | Role |
|---|---|---|
| **Product agents** | `app/agent/` (to implement per specs) | Runtime Research / Verifier / Supervisor that serve `/research` |
| **Cursor subagents** | `.cursor/agents/` | Coding assistants that implement, review, or eval that code |

Product agents are what graders run. Cursor subagents help *build* them.

## Non-negotiables

1. **Do not “agentify” by looping classic RAG.** Wrapping `/rag/query` in a `for` loop fails the feature bar. The model must evaluate intermediate results and choose search again / verify / clarify / finalize.
2. **Hard stop always.** Every agentic loop has `max_iterations` and/or `max_tool_calls`. Never unbounded.
3. **Keep `/chat` and `/rag/*` working.** Verified research is additive (`POST /research`), not a rewrite of the W15 baseline.
4. **No confident answers from failed tools.** On KB errors, timeouts, or malformed retrieval, acknowledge failure — do not invent corpus facts.
5. **Specs are source of truth for the extension.** If README and `docs/SPECS.md` drift, fix docs to match code after implementing — don’t invent a third design.
6. **Match existing style.** Python 3.11, FastAPI, Pydantic v2, DI via `Depends` / lifespan, tools via `ToolRegistry`. Prefer extending `LLMProvider` / orchestrator patterns over new frameworks (no LangGraph/Crew unless the user explicitly asks).
7. **Prompt revisions are evidence-driven.** New `prompts/prompt_vN` must cite a specific failure from prior MLflow traces in `prompts/CHANGELOG.md` — no speculative prompt churn.
8. **Prefer `uv` once Phase 10 lands.** Use `uv sync` / `uv run`; do not reintroduce unpinned `pip install -r` as the primary path.

## Architecture (current + target)

**Today:** Client → routers → `AgentOrchestrator` ↔ tools ↔ LLM providers; RAG → Qdrant.

**Target (specs):** Client → `POST /research` → Supervisor → Research ↔ Verifier → answer + step traces + tokens; traces/metrics → MLflow; golden regression → Evidently; optional Airflow nightly DAG.

Details: [`docs/architecture.md`](docs/architecture.md), [`docs/SPECS.md`](docs/SPECS.md).

## Where things live

| Path | Purpose |
|---|---|
| `docs/SPECS.md` | Phased build plan (Part I agentic + Part II MLOps) |
| `docs/architecture.md` / `.svg` | Architecture diagram (update when agentic + MLOps land) |
| `app/agent/` | Orchestrator today; research / verifier / supervisor next |
| `app/tools/` | Tool definitions + handlers; register new tools here |
| `app/llm/` | Provider ABC + Anthropic + OpenAI-compatible (incl. vLLM) |
| `app/rag/` | Chunk → embed → Qdrant → retrieve |
| `app/api/` | HTTP surface |
| `skills/` | **Runtime** progressive-disclosure skills loaded by the product (`load_skill`) |
| `prompts/` | Versioned system prompts (`prompt_v1.md` …) + CHANGELOG |
| `eval/` | Custom evaluation harness + golden set for Evidently |
| `mlops/` | MLflow tracking helpers, Evidently runner, Airflow DAG, reports |
| `tests/` | Unit tests; reuse `ScriptedProvider` patterns |
| `pyproject.toml` / `uv.lock` | Reproducible env (after Phase 10) |
| `.cursor/skills/` | Cursor coding skills |
| `.cursor/agents/` | Cursor subagents |

## Tech stack

Python 3.11 · FastAPI · Pydantic Settings · Anthropic SDK · OpenAI SDK · Qdrant · sentence-transformers · Docker Compose · pytest · **uv** · **MLflow** · **Evidently** · **Airflow** (optional runtime)

## How to work

1. Read the relevant phase in [`docs/SPECS.md`](docs/SPECS.md) before coding.
2. Prefer the Cursor skill that matches the task (see below) instead of inventing a parallel workflow.
3. For large or parallel work, delegate to a `.cursor/agents/` subagent.
4. Run `make test` (or `pytest -v`) after behavioral changes. Use `make eval` once the harness exists.
5. Do not commit unless the user asks.

## Cursor skills (project)

| Skill | Use when |
|---|---|
| `implement-verified-research` | Implementing supervisor / research / verifier / `/research` per specs |
| `context-engineering` | Adding Skills progressive disclosure, evidence notes, or tool-result capping |
| `eval-harness` | Building or running the from-scratch eval harness and failure taxonomy |
| `mlops-tracking` | uv migration, MLflow prompt experiments, Evidently regression, Airflow DAG |

Invoke with `/skill-name` or by doing work that matches the skill description.

## Cursor subagents (project)

| Subagent | Use when |
|---|---|
| `spec-implementer` | Implementing one or more phases from `docs/SPECS.md` end-to-end |
| `agent-loop-reviewer` | Reviewing stop conditions, branching, and “not a fixed pipeline” compliance |
| `eval-runner` | Running the eval harness and summarizing metrics / failures |
| `mlops-runner` | Running MLflow experiment loops, Evidently suites, exporting reports |
| `docs-sync` | Updating README (Track B + MLOps a–d) and architecture diagram to match code |

## Runtime product skill

| Skill path | Role |
|---|---|
| `skills/verified_research/SKILL.md` | Loaded by the Research agent via progressive disclosure (context engineering) |

That file is **product context**, not a Cursor coding skill. Do not move it under `.cursor/skills/`.

## Nested guidance

- [`app/agent/AGENTS.md`](app/agent/AGENTS.md) — rules for product agent loops and context budgets.
