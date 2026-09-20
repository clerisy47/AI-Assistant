"""Normalize supervisor tool_trace entries into Phase 11 step-trace schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_step(entry: dict[str, Any], step: int) -> dict[str, Any]:
    """Map a raw tool_trace dict to the SPECS Phase 11 step object."""
    args = entry.get("args")
    if args is None:
        args = entry.get("arguments", {})
    agent = str(entry.get("agent") or "research")
    tool = entry.get("tool")
    is_error = bool(entry.get("is_error", False))

    decision = entry.get("decision")
    if decision is None and tool:
        if tool == "search_knowledge_base":
            decision = "search"
        elif tool == "ask_clarification":
            decision = "clarify"
        elif tool == "verify_claims":
            decision = "verify"
        elif tool == "update_evidence_notes":
            decision = "record_notes"
        elif tool == "load_skill":
            decision = "load_skill"
        else:
            decision = "tool_call"

    out: dict[str, Any] = {
        "step": step,
        "agent": agent,
        "decision": decision,
        "reasoning": entry.get("reasoning"),
        "tool": tool,
        "args": args if isinstance(args, dict) else {"value": args},
        "result": _cap_result(entry.get("result")),
        "is_error": is_error,
    }
    return out


def _cap_result(result: Any, max_chars: int = 4000) -> Any:
    if result is None:
        return None
    if isinstance(result, (dict, list)):
        text = json.dumps(result, ensure_ascii=False)
    else:
        text = str(result)
    if len(text) <= max_chars:
        return text if not isinstance(result, (dict, list)) else result
    return text[: max_chars - 3] + "..."


def normalize_trace(
    tool_trace: list[dict[str, Any]] | None,
    *,
    start_step: int = 1,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for i, entry in enumerate(tool_trace or []):
        if not isinstance(entry, dict):
            continue
        steps.append(normalize_step(entry, start_step + i))
    return steps


def build_run_footer(
    *,
    iterations: int,
    stop_reason: str,
    token_usage: dict[str, Any] | None,
    prompt_version: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "run_footer",
        "iterations": iterations,
        "stop_reason": stop_reason,
        "token_usage": token_usage or {},
        "prompt_version": prompt_version,
        "config": config or {},
    }


def write_trace_jsonl(
    path: Path,
    *,
    tool_trace: list[dict[str, Any]] | None,
    iterations: int,
    stop_reason: str,
    token_usage: dict[str, Any] | None,
    prompt_version: str,
    config: dict[str, Any] | None = None,
) -> Path:
    """Write one JSON object per line (steps) plus a final footer line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = normalize_trace(tool_trace)
    footer = build_run_footer(
        iterations=iterations,
        stop_reason=stop_reason,
        token_usage=token_usage,
        prompt_version=prompt_version,
        config=config,
    )
    lines = [json.dumps(s, ensure_ascii=False) for s in steps]
    lines.append(json.dumps(footer, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
