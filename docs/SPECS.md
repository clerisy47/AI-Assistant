# Specs — Verified Research Agent + MLOps

**Status:** In progress — Phases 0–10 complete; next is Phase 11 (MLflow experiments)  
**Base:** Existing W15 assistant (`POST /chat` tool loop, RAG-as-tool, FastAPI + Qdrant)  
**Tracks covered:**
- **Track B (Agentic AI):** Cross-source verified research with a multi-agent loop, context engineering, and a custom eval harness (Phases 0–9).
- **Track A (Tool Data Science / MLOps):** `uv` env, MLflow experiment tracking on prompts/config + full agent traces, Evidently LLM regression tests, optional Airflow scheduled eval (Phases 10–14).

**Goal:** Extend the assistant with an agentic feature *and* apply MLOps disciplines to configurations and behavior (prompts, retrieval settings, agent parameters, harness metrics) — not to a trained model.  
**Orientation for coding agents:** see root [`AGENTS.md`](../AGENTS.md). When a phase’s acceptance criteria are met, update this file’s checkboxes and status (see AGENTS.md → “Updating SPECS after a phase”).

---

## Part I — Agentic feature (Track B)


### Chosen feature: Cross-source verified research

**One-sentence justification (assignment requirement):**  
A fixed pipeline cannot decide *whether* evidence from the first retrieval is sufficient, *which* follow-up query to run when it is not, or *whether* a draft answer is supported by sources — those decisions depend on intermediate results only the model can evaluate.

### What the user experiences

New endpoint (or mode) that answers research-style questions by:

1. Searching the knowledge base (possibly multiple times with refined queries).
2. Optionally comparing / gathering evidence across distinct source files or query angles.
3. Drafting an answer, then verifying claims against retrieved evidence.
4. Revising or searching again if verification fails.
5. Stopping when evidence is sufficient **or** a hard iteration / step budget is hit — never unbounded.

Example queries this feature should handle well:

| Query type | Why it needs an agentic loop |
|---|---|
| “What does this assistant support for local LLMs, and is that consistent across docs?” | May need multiple KB searches + consistency check |
| “Compare calculator vs knowledge-base usage guidance in the corpus” | Open-ended; tool order depends on what turns up |
| “Summarize RAG ingestion and confirm chunking claims against sources” | Draft → verify → revise path |

### Explicit non-goals

- Do **not** put the existing `/rag/query` retrieve-then-generate path inside a `for` loop and call that “agentic.”
- Do **not** require a real web search for MVP (corpus-only is fine); web can be a later tool if desired.
- Do **not** replace the current simple `/chat` loop — keep it; add a distinct verified-research path.

---

## 1. Design decisions (write these into README later)

These answers satisfy the assignment’s documentation / additional requirements. Implement so the README can cite real code paths.

### 1.1 Why not a Skill? (Skill vs Agent)

**Sentence for README:**  
Progressive disclosure of research procedures *could* be a Skill (load `SKILL.md` only when relevant), but the capability itself is an **agent** because it must iteratively call tools, evaluate intermediate evidence, and branch — a Skill is procedural context, not a runtime decision loop.

Use a Skill **in addition** for context engineering (see §1.2), not instead of the agent.

### 1.2 Context engineering technique

| Field | Decision |
|---|---|
| **Technique** | Progressive disclosure via Skills **plus** capping/clearing verbose tool results |
| **Where in the loop** | (1) Start with a short `skills/verified_research/SKILL.md` summary in system context; load full skill body only when the model selects `load_skill("verified_research")` or the verified-research mode is entered. (2) After each `search_knowledge_base` call, cap returned chunks (e.g. top-k + char budget) and optionally replace older raw tool payloads with compact summaries once verification notes exist. |
| **Problem solved** | Full research instructions + multi-turn retrieval dumps saturate context quickly (context saturation), which degrades later tool-choice quality and burns tokens in eval. |

### 1.3 Agentic pattern

| Field | Decision |
|---|---|
| **Pattern** | **Multi-agent system**: Research agent + Verifier agent, coordinated by a thin supervisor loop |
| **Why** | Avoids the **self-verification paradox** (same agent that wrote the answer rubber-stamps it) and provides **context isolation** (verifier sees draft + evidence notes, not the full exploratory tool dump) and **specialization** (research tools vs verify-only tools) |
| **Also acceptable fallback** | If multi-agent proves too costly in tokens, document a single-agent loop with a `verify_against_sources` tool and cite eval token comparison — but **default plan is multi-agent** |

Structural-failure framing (for README §b):

- Multi-agent addresses **self-verification paradox** and **context saturation**.
- We accept some **coordination cost** (extra tokens); the eval harness must make that visible vs a single-agent baseline.

### 1.4 Tool vs agent boundary

**Paragraph for README:**  
Qdrant retrieval and the LLM provider are modeled as **bounded tool / provider calls**, not peer agents. Retrieval is a single-shot request/response (`search_knowledge_base`); the provider is a synchronous generate step. Stateful multi-step behavior (decide next query, decide whether to verify, revise) lives in *our* agent loop. Treating Qdrant or the cloud LLM API as agents would add coordination overhead without isolation or specialization benefits — they are external services with fixed contracts, not collaborators that need their own goals.

### 1.5 Stopping conditions (hard requirements)

The loop **must** stop when any of these hold:

1. Verifier reports `sufficient=true` and research agent emits a final answer, **or**
2. Research agent explicitly asks the user for clarification (and returns that as the response), **or**
3. `max_iterations` (default: 8 supervisor steps) is reached, **or**
4. `max_tool_calls` (default: 12) across both agents is reached.

Never run indefinitely. On budget exhaustion, return a clear partial answer + `tool_trace` + `stop_reason`.

---

## 2. Target architecture

```text
Client
  │
  ▼
POST /research  (or POST /chat with mode=verified_research)
  │
  ▼
Supervisor (iteration budget, stop conditions, token accounting, step trace writer)
  │
  ├─► Research Agent
  │     tools: search_knowledge_base, load_skill, ask_clarification,
  │            calculator, get_current_datetime (optional)
  │     writes: structured EvidenceNotes (external notes)
  │     emits: step records {step, tool, args, result, reasoning}
  │
  └─► Verifier Agent  (separate context)
        inputs: draft answer + EvidenceNotes (not full raw tool dump)
        tools: check_claims_against_notes (or read-only re-search capped)
        outputs: {sufficient, issues[], revised_hints[]}

Context engineering:
  - SKILL.md progressive disclosure
  - Cap / clear raw retrieval text after notes are written

MLOps (Part II):
  - Prompt versions under prompts/
  - Each research run → MLflow params/metrics + step traces as artifacts
  - Evidently Test Suite vs golden references → pct_tests_passed logged to MLflow
  - Optional Airflow DAG: nightly harness + Evidently + degrade alert
```

Update `docs/architecture.md` / `.svg` in Phase 8 / 14 to show the agentic loop, multi-agent coordination, **and** the MLOps tracking/monitoring path.

---

## 3. Phased implementation plan

Each phase has: goal, deliverables, acceptance criteria, and suggested file touchpoints. Phases are sequential unless noted.

### Implementation progress

| Phase | Name | Status | Evidence (when DONE) |
|------:|------|--------|----------------------|
| 0 | Spec lock & baseline inventory | DONE | This file + root `AGENTS.md` |
| 1 | Evidence notes & context utilities | DONE | `app/agent/evidence_notes.py`, `context_budget.py`, `tests/test_evidence_notes.py` |
| 2 | Skills progressive disclosure | DONE | `skills/verified_research/SKILL.md`, `app/tools/skill_tool.py`, `tests/test_skill_tool.py` |
| 3 | Research agent loop | DONE | `app/agent/research_agent.py`, `app/tools/research_tools.py`, `tests/test_research_agent.py` |
| 4 | Verifier + supervisor + API | DONE | `app/agent/verifier_agent.py`, `app/agent/supervisor.py`, `app/api/research.py`, `tests/test_supervisor.py` |
| 5 | Token / cost accounting | DONE | `app/llm/usage.py`, providers + `StructuredLLMResponse`, supervisor rollup, `baseline=single`, `tests/test_usage.py` |
| 6 | Eval harness | DONE | `eval/harness.py`, `cases.yaml`, `report.md`, `tests/test_eval_scoring.py` |
| 7 | Failure injection | DONE | `INJECT_FAILURE` in config, `knowledge_base_tool.py` inject hook, supervisor `tool_failure`, `eval` case `kb_unavailable_recognized`, `tests/test_failure_injection.py` |
| 8 | Track B docs + architecture | DONE | `README.md` Track B a–c + additional; `docs/architecture.md` / `.svg` (supervisor + research↔verifier; MLOps stubbed) |
| 9 | Polish / Track B checklist | DONE | `sample_docs/{about_this_assistant,tools_and_usage,rag_pipeline}.md`; README `/research` demo; `eval/report.md`; §8 Track B checked |
| 10 | Environment (`uv`) | DONE | `pyproject.toml`, `uv.lock`, Makefile/Dockerfile/`uv sync --frozen` |
| 11 | MLflow experiments | NOT STARTED | |
| 12 | Evidently regression | NOT STARTED | |
| 13 | Airflow DAG | NOT STARTED | |
| 14 | MLOps docs polish | NOT STARTED | |

Coding agents: when you complete a phase, mark its deliverables `- [x]` below, set this table’s Status to `DONE` (or `PARTIAL`), and refresh the header `**Status:**` line.

---

### Phase 0 — Spec lock & baseline inventory

**Goal:** Confirm the feature sentence, pattern, and context technique; inventory current hooks.

**Deliverables**

- [x] This specs file reviewed / adjusted if needed
- [x] One-sentence feature justification finalized (§0)
- [x] Note current baseline: `AgentOrchestrator` in `app/agent/orchestrator.py`, tools in `app/tools/`, eval gap (unit tests only)

**Acceptance**

- Team can explain in one sentence why a fixed pipeline fails for this feature.
- Skill vs agent sentence drafted (§1.1).

**Estimated effort:** &lt; 1 hour (docs only)

---

### Phase 1 — Evidence notes schema & context utilities

**Goal:** Introduce structured external notes and retrieval capping — the scaffolding for context engineering.

**Deliverables**

- [x] `app/agent/evidence_notes.py` — Pydantic models, e.g.:

  ```python
  class EvidenceItem(BaseModel):
      source: str
      excerpt: str
      query_used: str
      relevance: float | None = None

  class EvidenceNotes(BaseModel):
      question: str
      items: list[EvidenceItem]
      open_gaps: list[str] = []
      sufficient_hypothesis: bool = False
  ```

- [x] `app/agent/context_budget.py` — helpers:
  - `cap_tool_result(text, max_chars)`
  - `compact_messages(messages, keep_last_n_tool_results)` optional
- [x] Unit tests for capping and note serialization

**Acceptance**

- Notes round-trip JSON cleanly.
- Cap never returns unbounded retrieval dumps into the verifier context.

**Suggested paths:** `app/agent/`, `tests/test_evidence_notes.py`

---

### Phase 2 — Skills progressive disclosure

**Goal:** Add Skill loading as the documented context-engineering technique.

**Deliverables**

- [x] `skills/verified_research/SKILL.md` — short front-matter / description (~10–20 lines) + full procedure body (**stub already present**; refine during implementation)
- [x] Tool `load_skill(name: str)` registered only for the research agent
- [x] Initial system prompt includes **only** skill titles + one-line descriptions; full body injected after `load_skill`
- [x] Unit test: without load, full body absent from messages; after load, present once

**Acceptance**

- README later can point to exact files for “progressive disclosure via Skills.”
- Skill content is not dumped into every `/chat` request — only the verified-research path.

**Suggested paths:** `skills/verified_research/SKILL.md`, `app/tools/skill_tool.py`

**Cursor helpers:** skill `context-engineering`; subagent `spec-implementer`

---

### Phase 3 — Research agent loop (decision-capable)

**Goal:** Research agent that can iterate: search → evaluate sufficiency → search again / clarify / draft.

**Deliverables**

- [x] `app/agent/research_agent.py` (or extend orchestrator with a dedicated class)
- [x] Tools available:
  - `search_knowledge_base` (existing; results capped)
  - `load_skill`
  - `update_evidence_notes` (write structured notes)
  - `ask_clarification` (ends turn with a question to the user)
  - Existing builtins optional
- [x] Loop allows **&gt; 1 iteration** per request; model chooses next action from tool results
- [x] Hard stop: `max_iterations` / `max_tool_calls`
- [x] Returns: `{draft_answer, evidence_notes, tool_trace, iterations, token_usage}`

**Acceptance**

- Scripted LLM tests show: first search insufficient → second search with refined query → draft.
- Never exceeds max iterations.
- Fixed single-pass RAG path remains unchanged at `/rag/query`.

**Suggested paths:** `app/agent/research_agent.py`, `tests/test_research_agent.py`

---

### Phase 4 — Verifier agent + supervisor (multi-agent)

**Goal:** Separate verifier context; supervisor coordinates research ↔ verify until stop.

**Deliverables**

- [x] `app/agent/verifier_agent.py`
  - Input: user question + draft + `EvidenceNotes` (not full exploratory transcript)
  - Output: structured `VerificationResult` (`sufficient`, `unsupported_claims`, `suggested_next_action`)
- [x] `app/agent/supervisor.py`
  - Alternates or routes: research → verify → (re-research | finalize | clarify)
  - Enforces global budgets
  - Aggregates `tool_trace` and token usage from both agents
- [x] API: `POST /research` (preferred) **or** `ChatRequest.mode = "verified_research"`
- [x] Response schema includes: `answer`, `verification`, `tool_trace`, `iterations`, `stop_reason`, `token_usage`

**Acceptance**

- Verifier does not receive raw multi-search dumps by default (context isolation).
- At least one integration-style scripted test: verify fails → research runs again → verify passes.
- Stopping conditions from §1.5 enforced in code.

**Suggested paths:** `app/agent/supervisor.py`, `app/api/research.py`, `app/schemas/research.py`

---

### Phase 5 — Token / cost accounting

**Goal:** Every research query records token consumption; multi-agent vs single-agent baseline comparable.

**Deliverables**

- [x] Extend LLM provider responses (or wrap `generate`) to expose `prompt_tokens`, `completion_tokens`, `total_tokens` when the API returns them; estimate if missing (document estimation method)
- [x] Per-query totals in API response and eval harness
- [x] Optional: `POST /research?baseline=single` that runs a single-agent variant for comparison (same tools, no separate verifier) — used by eval only

**Acceptance**

- Eval results table includes a Tokens column per query.
- Multi-agent vs single-agent token delta visible when baseline is run.

**Suggested paths:** `app/llm/base.py`, providers, `app/agent/supervisor.py`

**Notes (implemented):** Estimation uses chars÷4 when the API omits usage (`app/llm/usage.py`). Baseline is request body `baseline: "single"` (not query string). `generate_structured` returns `StructuredLLMResponse` with usage. Eval Tokens column lands with the Phase 6 harness; `/research` already returns `token_usage`.

---

### Phase 6 — Evaluation harness (from scratch)

**Goal:** Custom harness — no LangSmith / RAGAS / DeepEval / etc. — measuring the new agentic feature.

**Deliverables**

- [x] `eval/harness.py` — runner that:
  - Loads `eval/cases.yaml` (or `.json`) of research queries + expectations
  - Invokes supervisor (real or scripted provider for CI)
  - Records metrics below
- [x] `eval/report.md` generated (or printed Markdown table)
- [x] Metrics (minimum):

  | Metric | Definition |
  |---|---|
  | Task completion rate | Fraction of cases where `stop_reason` is success and answer meets case rubric (keyword / claim checklist) |
  | Tool-call correctness | Expected tools called with valid args (schema + allowlist of tool names / required args per case) |
  | Trajectory length | Iterations / tool calls per case; flag outliers vs `expected_max_iterations` |
  | Token usage | Total tokens per case; multi-agent vs single-agent when baseline enabled |
  | Failure log | Classify failures (see below) |

- [x] Failure taxonomy (class discussion mapping):

  | Class | Meaning in this system |
  |---|---|
  | **Hard failure** | Crash, empty answer on budget exhaustion with no partial, or unverifiable fabrication marked as final success |
  | **Soft failure** | Wrong tool once, weak answer, or extra iteration — task still completes with caveats |
  | **Cascading soft failure** | Early soft error (bad query / empty retrieval) causes later wrong verify or confident unsupported answer |

- [x] `make eval` or `python -m eval.harness` documented in README

**Acceptance**

- Running the harness produces a Markdown table + failure log without external eval frameworks.
- At least 8–12 cases covering: sufficient first search, needs re-search, clarification, verify-fail-then-revise, max-iteration stop.

**Suggested paths:** `eval/harness.py`, `eval/cases.yaml`, `eval/report.md`, `Makefile`

---

### Phase 7 — Failure injection test

**Goal:** Intentionally break one dependency and document agent behavior.

**Deliverables**

- [x] Configurable failure mode, e.g. env `INJECT_FAILURE=kb_unavailable` or harness flag:
  - Make `search_knowledge_base` return a structured error / timeout / malformed payload
- [x] One documented case in `eval/report.md` or README “Failure injection”
- [x] Assert desired behavior: agent **acknowledges** failure / asks to retry / refuses confident answer — does **not** invent corpus facts

**Acceptance**

- Write-up states what was injected and how the agent responded (recognize vs hallucinate).
- Automated test or harness case covers this path.

**Suggested paths:** `app/tools/knowledge_base_tool.py` (hook), `eval/cases.yaml`, README subsection

---

### Phase 8 — Documentation & architecture diagram (Track B)

**Goal:** Assignment deliverables for agentic track: README sections a–c + additional requirements (~one page), updated diagram (agentic loop). MLOps README a–d lands in Phase 14.

**Deliverables**

- [x] README sections (implementation-specific, not course lecture recap):

  - **a. Context Engineering Technique** — which, where in loop, what problem
  - **b. Agentic Pattern** — multi-agent vs single; why (isolation / specialization / self-verification paradox)
  - **c. Evaluation Harness** — how metrics are measured; point to `eval/`
  - **Additional requirements** — Skill vs Agent; token accounting; failure injection; tool vs agent boundary
  - Keep Track B a–c + additional ≈ **one page**

- [x] Update `docs/architecture.md` / `docs/architecture.svg` (and mermaid source if present):
  - Show supervisor + research ↔ verifier loop
  - Show skill load + evidence notes
  - Keep existing classic `/chat` and `/rag/query` paths visible
  - (Stub MLOps boxes if Phase 11+ not done yet; finalize in Phase 14)

- [x] Link specs → README → eval report for graders

**Acceptance**

- A grader can map each Track B assignment bullet to a file or README paragraph.
- Diagram shows agentic loop and multi-agent coordination.

---

### Phase 9 — Polish, regression, submission checklist

**Goal:** Ship-ready.

**Deliverables**

- [x] Existing unit tests still pass (`make test`)
- [x] New tests for research/verifier/supervisor/harness helpers
- [x] Sample docs sufficient for demo queries (extend `sample_docs/` if needed)
- [x] `.env.example` updated for any new settings (`MAX_RESEARCH_ITERATIONS`, `INJECT_FAILURE`, etc.)
- [x] Submission checklist completed (below)

**Acceptance**

- Clean demo path: ingest sample docs → `POST /research` → show trace + verification in Swagger.
- Eval harness run produces `eval/report.md`.

---

## 4. API sketch

### `POST /research`

```json
// Request
{
  "messages": [{"role": "user", "content": "Compare local LLM support claims in the docs."}],
  "provider": null,
  "temperature": 0.3,
  "max_iterations": 8
}

// Response
{
  "answer": "...",
  "verification": {
    "sufficient": true,
    "unsupported_claims": [],
    "notes": "All claims backed by evidence notes."
  },
  "evidence_notes": { "...": "..." },
  "tool_trace": [ /* research + verify tool calls */ ],
  "iterations": 4,
  "stop_reason": "verified",
  "token_usage": {
    "prompt_tokens": 12000,
    "completion_tokens": 900,
    "total_tokens": 12900,
    "by_agent": {"research": 10000, "verifier": 2900}
  }
}
```

`stop_reason` enum: `verified` | `clarification` | `max_iterations` | `max_tool_calls` | `tool_failure`

Keep `POST /chat` unchanged for backward compatibility.

---

## 5. Config additions

| Setting | Default | Purpose |
|---|---|---|
| `MAX_RESEARCH_ITERATIONS` | `8` | Supervisor research↔verify round budget |
| `MAX_RESEARCH_TOOL_CALLS` | `12` | Global tool-call budget across research passes |
| `MAX_RESEARCH_PASS_ITERATIONS` | `4` | LLM turns per research pass inside the supervisor |
| `EVIDENCE_EXCERPT_MAX_CHARS` | `500` | Cap per evidence item |
| `TOOL_RESULT_MAX_CHARS` | `2000` | Cap raw tool results in research context |
| `SKILLS_DIR` | `skills` | Progressive disclosure root |
| `INJECT_FAILURE` | empty | Failure injection for eval / demo |
| `PROMPT_VERSION` | `prompt_v1` | Active system prompt file under `prompts/` |
| `MLFLOW_TRACKING_URI` | `./mlruns` | Experiment tracking backend |
| `MLFLOW_EXPERIMENT_NAME` | `verified-research` | Experiment name |
| `EVIDENTLY_PASS_THRESHOLD` | `0.8` | Min `pct_tests_passed` to promote a version |
| `REGRESSION_DEGRADE_PP` | `10` | Alert if completion or pass rate drops by this many percentage points |
| `GOLDEN_SET_PATH` | `eval/golden_set.yaml` | Evidently reference set |

---

## 6. Testing strategy

| Layer | What | When |
|---|---|---|
| Unit | Notes, capping, skill load, stop conditions, tool schema validation | Every phase |
| Scripted LLM | Fake provider sequences for research → verify → revise | Phases 3–4 |
| Harness | Case suite + metrics + failure injection | Phases 6–7 |
| MLflow runs | ≥3 prompt/config versions with full step traces | Phase 11 |
| Evidently | LLM-as-judge regression vs golden set | Phase 12 |
| Airflow (optional) | Scheduled harness + degrade flag | Phase 13 |
| Manual demo | Live Anthropic/OpenAI + Qdrant with sample docs | Before submit |

Reuse the existing `ScriptedProvider` pattern from `tests/test_agent_orchestrator.py`.

---

## Part II — MLOps (Track A)

There is no trained model to version. Tracking and monitoring apply to **prompts, retrieval settings, agent parameters, and eval metrics** from the Part I harness.

### MLOps design decisions (README sections a–d later)

| Discipline | Decision for this repo |
|---|---|
| **uv** | Replace `requirements.txt` / `requirements-dev.txt` with `pyproject.toml` + committed `uv.lock`. One-command: `uv sync` then `uv run …`. |
| **MLflow** | Experiment on versioned system prompts (`prompts/prompt_v1.md` …) plus config (chunk size, top-k, model, temperature, `max_iterations`). Each run logs params, harness metrics, Evidently `pct_tests_passed`, and **full step traces** (tool args/results + intermediate decisions + stop reason). ≥3 versions; each revision driven by a **specific failure seen in prior traces**. |
| **Evidently** | Fixed golden query→reference set from best earlier version. After each prompt/config change, run Test Suite with LLM-as-judge (`BinaryClassificationPromptTemplate` or equivalent) for reference-based correctness (+ at least one more check). Failures = real regressions; log pass rate to MLflow. |
| **Airflow** | Optional DAG: nightly (or cron) regression eval; alert/flag if harness or Evidently metrics degrade past a threshold. |

---

### Phase 10 — Environment management (`uv`)

**Goal:** Reproducible Python env for the assistant; kill “works on my machine” dependency drift.

**Why uv (project-specific):** Before Phase 10, pins lived in loose `requirements*.txt` without a lockfile; Docker and local `pip install` could resolve different transitive versions (FastAPI/Pydantic/qdrant-client/sentence-transformers), which silently changed RAG and tool-loop behavior between machines and broke MLflow run comparability.

**Deliverables**

- [x] `pyproject.toml` with project metadata, runtime deps, optional extras (`[dev]`, `[mlops]` for mlflow/evidently; Airflow deferred to Phase 13)
- [x] Committed `uv.lock`
- [x] Migrate Makefile / Dockerfile / README to `uv sync` / `uv run`
- [x] Docker path: **`uv sync --frozen` in the builder** (same lockfile as local; no `uv export` → pip) — documented in README + Dockerfile comments
- [x] Confirm clean-clone path: `uv sync && uv run pytest` (and later `uv run python -m eval.harness`)

**Acceptance**

- Fresh clone + `uv sync` installs a locked env without hand-editing requirements.
- README §a documents the one-command reproduction path and the concrete dependency problem uv solves here.

**Suggested paths:** `pyproject.toml`, `uv.lock`, `Makefile`, `Dockerfile`, `.python-version`

**Notes (implemented):** `tool.uv.package = false` (app imported from repo root). `[mlops]` declares mlflow + evidently only; Airflow lands in Phase 13.

---

### Phase 11 — Experiment tracking (MLflow)

**Goal:** Version prompts/config, run the same test queries, capture **full agent traces** (not just final answers), iterate ≥3 versions from trace diagnoses, compare in MLflow.

#### Prompt versioning

Store explicit versions as separate files (preferred):

```text
prompts/
  prompt_v1.md          # baseline research system prompt
  prompt_v2.md          # revision driven by v1 failure (cite which)
  prompt_v3.md          # revision driven by v2 failure (cite which)
  CHANGELOG.md          # one paragraph per version: failure seen → change made
```

Also version (as MLflow params) any of: `chunk_size`, `top_k`, `model`, `temperature`, `max_iterations`, `max_tool_calls`, single vs multi-agent flag.

#### Step trace schema (required)

Agentic loop must emit structured steps — one JSON object per step — e.g.:

```json
{
  "step": 3,
  "agent": "research",
  "decision": "search_again",
  "reasoning": "First hit lacked local-LLM details; refining query.",
  "tool": "search_knowledge_base",
  "args": {"query": "local vLLM Llama deployment"},
  "result": "...capped...",
  "is_error": false
}
```

Plus a run-level footer: `iterations`, `stop_reason` (`verified` | `clarification` | `max_iterations` | `max_tool_calls` | `tool_failure`), token totals.

Implementation options (pick one and document):

1. **Custom:** supervisor writes `traces/<run_id>/<query_id>.jsonl` then `mlflow.log_artifact`.
2. **MLflow tracing:** `mlflow.trace` / LLM autolog so traces appear in the MLflow UI next to params/metrics.

#### Per-version run protocol

For each `prompt_vN` (+ config):

1. Start an MLflow run tagged with `prompt_version`, config params.
2. Execute fixed test queries through `/research` (or harness entrypoint).
3. Log metrics from Part I harness (completion rate, tool-call correctness, mean trajectory length, mean tokens, failure counts).
4. Save **≥2–3 representative traces as artifacts** per version — ideally **one clean success + one failure** (failure informs the next prompt revision).
5. Log prompt file + harness report as artifacts.
6. **Diagnose** underperformance from traces (stopped searching too early, misread tool result, wrong tool, etc.) and write that diagnosis into `prompts/CHANGELOG.md` before authoring `vN+1`.

**Hard rule:** Each new prompt version must respond to a **specific failure observed in traces**, not a speculative tweak.

#### Minimum experiment budget

- [ ] At least **3** prompt/configuration versions run end-to-end
- [ ] MLflow comparison (UI export or Markdown table under `mlops/reports/`) showing which version wins and the trade-off (e.g. completion ↑ vs tokens ↑)
- [ ] Exported run comparison checked into repo or documented path to reload (`mlruns/` or remote tracking URI)

**Deliverables**

- [ ] `mlops/tracking.py` (or `eval/mlflow_runner.py`) — start run, log params/metrics/artifacts
- [ ] Step-trace writer hooked in supervisor / orchestrator
- [ ] `prompts/prompt_v{1,2,3}.md` + `CHANGELOG.md`
- [ ] `mlops/reports/mlflow_comparison.md` (or HTML/CSV export)
- [ ] Config: `MLFLOW_TRACKING_URI` (default `./mlruns`)

**Acceptance**

- Opening MLflow UI shows ≥3 runs with params, metrics, and trace artifacts.
- CHANGELOG links each `vN` → prior failure case.
- README §b states what varied, what was measured, which config won, and the trade-off.

**Suggested paths:** `prompts/`, `mlops/tracking.py`, `mlops/reports/`, extend `app/agent/supervisor.py`

---

### Phase 12 — Monitoring & regression testing (Evidently AI)

**Goal:** Detect prompt/config regressions with a fixed golden set and LLM-as-a-judge; treat failures as real regressions; log pass rate into MLflow.

**Deliverables**

- [ ] `eval/golden_set.yaml` (or JSON) — fixed representative queries + approved **reference** answers (from best Part I / earlier prompt version)
- [ ] `mlops/evidently_regression.py` — build current responses for the same queries; run Evidently Test Suite with `evidently[llm]`
- [ ] **At least two judge checks**, including:
  1. **Reference-based correctness** — does the new response contradict or lose information present in the reference? (Evidently `BinaryClassificationPromptTemplate` → `correct`/`incorrect` + reasoning)
  2. **Second check** (choose and document one): e.g. faithfulness to retrieved evidence / refusal quality on tool-failure cases / no fabricated citations
- [ ] HTML report written to `mlops/reports/evidently_<prompt_version>.html`
- [ ] Log `pct_tests_passed` (and optionally per-check rates) as **MLflow metrics** on the same run as Phase 11
- [ ] Promotion rule: if too many cases fail (threshold e.g. `pct_tests_passed < 0.8`), **do not promote** that prompt version
- [ ] Short interpretation note: which cases failed, whether judge verdicts match human reading (sanity-check the judge)

**Reference vs current (for README §c)**

| Term | Meaning here |
|---|---|
| **Reference** | Golden answers from the approved baseline prompt/config |
| **Current** | Answers from the candidate prompt/config under test |

**Acceptance**

- Re-running after a deliberate bad prompt change fails the Test Suite.
- Pass/fail shows up beside harness metrics in MLflow.
- README §c explains reference/current, metrics, report takeaway, and action on threshold breach.

**Suggested paths:** `eval/golden_set.yaml`, `mlops/evidently_regression.py`, `mlops/reports/*.html`

---

### Phase 13 — Orchestration (Airflow) — optional but specified

**Goal:** Scheduled regression eval that flags metric degradation.

**Deliverables**

- [ ] Airflow DAG under `mlops/airflow/dags/regression_eval_dag.py` (or `dags/`)
- [ ] Schedule: e.g. `@daily` / cron `0 2 * * *` (document chosen schedule)
- [ ] Tasks (suggested):
  1. Ensure env / pull latest prompts config pointer
  2. Run eval harness (`uv run python -m eval.harness`)
  3. Run Evidently regression suite
  4. Log metrics to MLflow
  5. **Branch on degradation:** if `task_completion_rate` or `pct_tests_passed` drops more than threshold vs last promoted run → mark run failed / write alert artifact / optional webhook stub
- [ ] Document local how-to: `astro` / `airflow standalone` / docker-compose profile `airflow` — pick one and keep it runnable
- [ ] If Airflow is too heavy for the grader’s machine, still ship the DAG code + a `make airflow-dry-run` that executes the same Python callables without a full cluster

**Acceptance**

- README §d describes schedule, trigger condition, and behavior on positive drift/degradation.
- DAG code is present and the callable pipeline runs outside Airflow for demos.

**Suggested paths:** `mlops/airflow/dags/regression_eval_dag.py`, `docker-compose.yml` profile or docs-only dry-run

---

### Phase 14 — MLOps docs, exports, and repo polish

**Goal:** Assessment-ready documentation, organization, and exported artifacts.

**Deliverables**

- [ ] README sections **a–d** (implementation-specific, not tool tutorials):
  - **a. Environment & Reproducibility (uv)**
  - **b. Experiment Tracking Strategy (MLflow)**
  - **c. Monitoring & Drift Strategy (Evidently)**
  - **d. Orchestration (Airflow)**
- [ ] Architecture diagram updated with MLflow / Evidently / Airflow boxes
- [ ] Repo layout tidy: `prompts/`, `mlops/`, `eval/`, `docs/`, clear setup in README
- [ ] Exported MLflow comparison + Evidently HTML reports committed under `mlops/reports/` (or documented generation commands + sample outputs)
- [ ] `.env.example` updated (`MLFLOW_TRACKING_URI`, Evidently/judge model keys, thresholds)
- [ ] Submission completeness pass: organization, docs quality, runnable paths from clean clone

**Acceptance**

- Grader can follow README alone: sync env → run app → run eval → view MLflow → open Evidently HTML → understand DAG.
- Track A + Track B checklists both green.

---

## 7. Deliverables mapping (assignment ↔ repo)

| Assignment deliverable | Repo location |
|---|---|
| Updated source (W15 + agentic feature) | `app/agent/*`, `app/api/research.py`, `skills/`, tools |
| Updated README (Track B a–c + additional) | `README.md` |
| Updated README (Track A / MLOps a–d) | `README.md` |
| Updated architecture diagram | `docs/architecture.md`, `docs/architecture.svg` |
| Evaluation harness + results | `eval/harness.py`, `eval/cases.yaml`, `eval/report.md` |
| `pyproject.toml` + `uv.lock` | repo root |
| Versioned prompts | `prompts/prompt_v*.md`, `prompts/CHANGELOG.md` |
| MLflow tracking + comparison export | `mlops/tracking.py`, `mlruns/` or remote URI, `mlops/reports/mlflow_comparison.md` |
| Evidently reports (HTML) | `mlops/reports/evidently_*.html` |
| Airflow DAG (optional track item) | `mlops/airflow/dags/` |
| Specs (this file) | `docs/SPECS.md` |
| Agent orientation | `AGENTS.md`, `app/agent/AGENTS.md` |
| Cursor skills / subagents | `.cursor/skills/`, `.cursor/agents/` |

---

## 8. Submission checklist

### Track B — Agentic

- [x] Feature sentence (why not fixed pipeline) in README
- [x] Skill vs Agent one-liner in README
- [x] Context engineering documented with *this* system’s problem/solution
- [x] Multi-agent (or justified single-agent) + framework language
- [x] Loop runs &gt;1 iteration; model decides next action; hard stop exists
- [x] Eval harness from scratch with all four metric groups + failure taxonomy
- [x] Token totals per query; multi-agent vs single-agent comparison if multi-agent
- [x] Failure injection documented with observed behavior
- [x] Tool vs agent boundary paragraph (Qdrant / LLM as tools)
- [x] Write-up ≈ one page for Track B required sections
- [x] Architecture diagram shows agentic loop (+ coordination if multi-agent)
- [x] `make test` and `make eval` (or documented commands) succeed

### Track A — MLOps

- [x] `pyproject.toml` + committed `uv.lock`; clean-clone `uv sync` works
- [ ] ≥3 prompt/config versions; each revision driven by a traced failure
- [ ] Full step traces logged (tool args/results + reasoning + stop reason)
- [ ] ≥2–3 representative trace artifacts per version (success + failure)
- [ ] MLflow params/metrics/artifacts + exported comparison; winner + trade-off documented
- [ ] Evidently golden set + Test Suite with ≥2 judge checks (incl. reference correctness)
- [ ] `pct_tests_passed` (or equivalent) logged to MLflow; failing suite blocks promotion
- [ ] Evidently HTML reports present; judge sanity-check noted in README
- [ ] Airflow DAG (or dry-run equivalent) + degradation threshold behavior documented
- [ ] README sections a–d (uv, MLflow, Evidently, orchestration) reflect **this** implementation
- [ ] Repo well-organized; setup/run/understand instructions clear for graders

---

## 9. Suggested implementation order (summary)

```text
Part I — Agentic (Track B)
Phase 0   Specs lock
Phase 1   Evidence notes + capping utilities
Phase 2   Skills progressive disclosure
Phase 3   Research agent (multi-iteration decisions)
Phase 4   Verifier + supervisor + /research API
Phase 5   Token accounting (+ optional single-agent baseline)
Phase 6   Evaluation harness + cases + report
Phase 7   Failure injection
Phase 8   README (Track B) + architecture diagram (agentic)
Phase 9   Polish & Track B checklist

Part II — MLOps (Track A)
Phase 10  uv migration (pyproject.toml + uv.lock)
Phase 11  MLflow: prompts v1–v3, step traces, comparison
Phase 12  Evidently regression + log pct_tests_passed to MLflow
Phase 13  Airflow scheduled regression DAG (optional runtime; ship DAG)
Phase 14  README (Track A a–d) + exports + final completeness
```

**Dependency note:** Phase 11–12 need the Part I harness and a step-trace-capable research loop (Phases 4–6). Phase 10 (`uv`) can start in parallel anytime after Phase 0.

---

## 10. Open choices (resolve before Phase 3 / 11 if possible)

### Agentic

1. **API surface:** dedicated `POST /research` (cleaner for grading) vs `mode` on `/chat`.
2. **Verifier tools:** notes-only verification vs allowing one capped re-search.
3. **Baseline:** always run single-agent twin in eval, or only when `COMPARE_BASELINE=1`.
4. **Provider for eval CI:** scripted-only in CI; live LLM optional locally.

**Recommendation:** `POST /research`, notes-only verifier, baseline behind flag, scripted CI + live local demo.

### MLOps

1. **Trace backend:** custom JSONL artifacts vs `mlflow.trace` autolog (custom is clearer for grading; autolog is nicer in UI — either is fine if documented).
2. **MLflow storage:** local `./mlruns` committed/exported vs remote tracking server.
3. **Evidently second check:** faithfulness vs citation presence vs tool-failure refusal quality.
4. **Airflow runtime:** full compose profile vs DAG-only + `make airflow-dry-run`.
5. **Promotion threshold:** default `pct_tests_passed < 0.8` and/or completion drop &gt; 10pp vs last promoted.

**Recommendation:** custom JSONL + `mlflow.log_artifact` (plus optional autolog), local `mlruns` + exported Markdown comparison, second Evidently check = faithfulness-to-notes on research answers, DAG + dry-run Makefile target, 0.8 pass threshold.
