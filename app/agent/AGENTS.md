# AGENTS.md — `app/agent/`

Rules for anyone editing product agent loops in this package.

## Scope

This directory owns the runtime decision loops:

- `orchestrator.py` — existing `/chat` tool loop (keep stable)
- `evidence_notes.py` / `context_budget.py` — structured notes + retrieval capping (Phase 1)
- Skill catalog + `load_skill`: `app/tools/skill_tool.py` (`build_skills_system_prefix`, `register_skill_tools`) — Phase 2; attached in research agent only
- `research_agent.py` — decision-capable research loop (Phase 3); tools via `app/tools/research_tools.py` (`build_research_registry`)
- `verifier_agent.py` — isolated draft + EvidenceNotes check via structured output (Phase 4)
- `supervisor.py` — research ↔ verify loop, global budgets, token rollup, optional `baseline=single` (Phase 4–5)
- `app/llm/usage.py` — normalize / estimate token usage (Phase 5)

## Rules

1. **Separate contexts for Research vs Verifier.** Do not pass the full exploratory tool dump into the verifier. Pass draft + `EvidenceNotes` (and optional capped re-search if specs allow).
2. **Every loop returns structured stop metadata:** `stop_reason`, `iterations`, `tool_trace`, and `token_usage`.
3. **Tool errors are data.** Surface them in the trace; let the model (or supervisor) decide to clarify/fail — never swallow into a fabricated answer.
4. **Budget before cleverness.** Enforce `MAX_RESEARCH_ITERATIONS` / `MAX_RESEARCH_TOOL_CALLS` in the **supervisor** (globals); per research pass use `MAX_RESEARCH_PASS_ITERATIONS` and remaining tool budget.
5. **Reuse `ToolRegistry` and `LLMProvider`.** Do not call vendor SDKs from agent modules.
6. **Tests:** Prefer scripted providers (see `tests/test_agent_orchestrator.py` / `tests/test_research_agent.py` / `tests/test_supervisor.py`) for branching paths: re-search, verify-fail-revise, clarify, max-iterations, tool failure.

## Skill vs agent reminder

`skills/verified_research/` is progressive-disclosure **context**. The loop that calls tools and branches is the **agent**. Do not replace the supervisor with a SKILL.md alone.
