"""Unit tests for progressive-disclosure skill loading."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from app.tools.builtin_tools import CALCULATOR_DEFINITION, DATETIME_DEFINITION, calculator, get_current_datetime
from app.tools.knowledge_base_tool import KNOWLEDGE_BASE_DEFINITION, make_knowledge_base_tool
from app.tools.registry import ToolRegistry
from app.tools.skill_tool import (
    LOAD_SKILL_DEFINITION,
    build_skills_system_prefix,
    discover_skills,
    load_skill,
    register_skill_tools,
    skill_catalog_section,
)

_CHAT_PY = Path(__file__).resolve().parents[1] / "app" / "api" / "chat.py"

# Distinctive phrases that live only in the full SKILL.md body (not frontmatter).
_FULL_BODY_MARKERS = (
    "Do not self-declare",
    "Handoff",
    "On verify failure",
    "Anti-patterns",
    "ask_clarification",
)


def test_discover_finds_verified_research():
    skills = discover_skills()
    names = [s.name for s in skills]
    assert "verified_research" in names
    vr = next(s for s in skills if s.name == "verified_research")
    assert "corpus-backed research" in vr.description.lower() or "research" in vr.description.lower()
    assert len(vr.description) > 10


def test_catalog_has_name_and_description_only():
    catalog = skill_catalog_section()
    assert "verified_research" in catalog
    assert "corpus-backed" in catalog.lower() or "research" in catalog.lower()
    for marker in _FULL_BODY_MARKERS:
        assert marker not in catalog, f"catalog leaked full-body phrase: {marker!r}"


def test_system_prefix_is_catalog_only_progressive_disclosure():
    prefix = build_skills_system_prefix()
    assert "Available skills" in prefix
    assert "load_skill" in prefix
    assert "verified_research" in prefix
    for marker in _FULL_BODY_MARKERS:
        assert marker not in prefix, f"system prefix leaked full-body phrase: {marker!r}"


def test_load_skill_returns_full_body():
    body = asyncio.run(load_skill("verified_research"))
    for marker in _FULL_BODY_MARKERS:
        assert marker in body


def test_load_skill_idempotent():
    first = asyncio.run(load_skill("verified_research"))
    second = asyncio.run(load_skill("verified_research"))
    assert first == second


def test_unknown_skill_is_error_via_registry():
    registry = ToolRegistry()
    register_skill_tools(registry)
    result, is_error = asyncio.run(registry.call("load_skill", {"name": "does_not_exist"}))
    assert is_error is True
    assert "unknown skill" in result.lower() or "does_not_exist" in result


def test_path_traversal_rejected():
    registry = ToolRegistry()
    register_skill_tools(registry)
    for bad in ("../something", "foo/bar", "verified_research/../../etc"):
        result, is_error = asyncio.run(registry.call("load_skill", {"name": bad}))
        assert is_error is True
        assert "invalid" in result.lower() or "error" in result.lower()


def test_load_skill_definition_is_strict():
    schema = LOAD_SKILL_DEFINITION.parameters
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())
    assert LOAD_SKILL_DEFINITION.name == "load_skill"


def test_chat_module_does_not_wire_load_skill():
    """`/chat` must stay free of progressive-disclosure skills (research-path only)."""
    chat_src = _CHAT_PY.read_text(encoding="utf-8")
    assert "load_skill" not in chat_src
    assert "register_skill_tools" not in chat_src
    assert "skill_tool" not in chat_src

    # Mirror chat._build_tool_registry without importing app.api.chat (pulls LLM SDKs).
    registry = ToolRegistry()
    registry.register(CALCULATOR_DEFINITION, calculator)
    registry.register(DATETIME_DEFINITION, get_current_datetime)
    registry.register(KNOWLEDGE_BASE_DEFINITION, make_knowledge_base_tool(MagicMock()))
    names = {d.name for d in registry.definitions()}
    assert "load_skill" not in names
    assert names == {"calculator", "get_current_datetime", "search_knowledge_base"}


def test_register_skill_tools_adds_load_skill():
    registry = ToolRegistry()
    register_skill_tools(registry)
    assert registry.has("load_skill")
    result, is_error = asyncio.run(registry.call("load_skill", {"name": "verified_research"}))
    assert is_error is False
    assert "Handoff" in result
