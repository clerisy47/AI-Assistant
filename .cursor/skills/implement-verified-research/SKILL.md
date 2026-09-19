---
name: implement-verified-research
description: >-
  Implements the verified-research multi-agent path (evidence notes, research
  agent, verifier, supervisor, POST /research) per docs/SPECS.md. Use when
  building agentic loop features, research/verifier agents, or the /research API.
---

# Implement Verified Research

## Before coding

1. Read [`docs/SPECS.md`](../../../docs/SPECS.md) for the target phase.
2. Skim root [`AGENTS.md`](../../../AGENTS.md) non-negotiables.
3. Do **not** wrap `/rag/query` in a loop.

## Phase map → files

| Phase | Create / touch |
|---|---|
| 1 | `app/agent/evidence_notes.py`, `app/agent/context_budget.py`, tests |
| 2 | `skills/verified_research/SKILL.md` (exists), `app/tools/skill_tool.py`, wire `load_skill` |
| 3 | `app/agent/research_agent.py`, register capped KB + notes tools |
| 4 | `app/agent/verifier_agent.py`, `app/agent/supervisor.py`, `app/api/research.py`, schemas |
| 5 | Token usage on provider responses + response field `token_usage` |
| 6–7 | `eval/` harness + failure injection hook |
| 8 | README sections a–c, `docs/architecture.*` |

## Implementation rules

- Extend `ToolRegistry`; keep tool JSON schemas strict (`additionalProperties: false`).
- Reuse `AgentOrchestrator` patterns or extract a shared loop helper — avoid a second LLM client stack.
- Supervisor enforces stop conditions from specs §1.5.
- Verifier context = draft + `EvidenceNotes` only (context isolation).
- Preserve `POST /chat` behavior and existing unit tests.

## Done when

- Scripted test shows: insufficient first search → second search → verify → finalize (or revise).
- Max-iteration path returns clear `stop_reason`.
- `make test` passes.
- `docs/SPECS.md` updated for the finished phase (checkboxes `- [x]`, progress table, header `**Status:**`) per root `AGENTS.md`.
