---
name: context-engineering
description: >-
  Applies progressive disclosure via Skills, evidence-note externalization, and
  tool-result capping for the verified-research agent loop. Use when reducing
  context saturation, adding load_skill, EvidenceNotes, or compacting retrieval dumps.
---

# Context Engineering

## Techniques this project uses

1. **Progressive disclosure (Skills)** — System prompt lists skill title + one line; full `skills/verified_research/SKILL.md` body enters context only after `load_skill("verified_research")`.
2. **Structured external notes** — Persist findings in `EvidenceNotes`; verifier reads notes, not raw multi-turn tool XML/JSON dumps.
3. **Cap / clear tool results** — `TOOL_RESULT_MAX_CHARS` / excerpt caps in `context_budget.py`; optionally drop older raw tool payloads once notes exist.

## Where it applies in the loop

```text
Supervisor
  → Research: may load_skill → search (capped) → update_evidence_notes
  → (optional) clear/compact old tool messages
  → Verifier: draft + EvidenceNotes only
```

## Problem → fix

| Problem | Fix |
|---|---|
| Full research playbook in every turn | Skill progressive disclosure |
| Multi-search dumps blow the window | Cap results + evidence notes |
| Verifier rubber-stamps from same noisy context | Separate agent + notes-only input |

## Checklist when adding retrieval tools

- [ ] Results pass through a character / top-k cap before messages.append
- [ ] Material facts land in `EvidenceNotes`, not only in ephemeral tool messages
- [ ] Verifier wiring does not receive the full research transcript by default
- [ ] README “Context Engineering Technique” cites the real modules
