"""Tests for versioned research prompt loading (Phase 11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.prompts import load_research_prompt, prompts_root, research_prompt_path
from app.agent.research_agent import ResearchAgent
from app.config import settings
from app.llm.base import LLMResponse
from app.tools.research_tools import build_research_registry
from app.agent.evidence_notes import EvidenceNotes


def test_load_prompt_v1_from_repo():
    text = load_research_prompt("prompt_v1")
    assert "corpus-backed research agent" in text
    assert "search_knowledge_base" in text


def test_load_unknown_version_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_research_prompt("prompt_v99_missing")


def test_invalid_version_name_rejected():
    with pytest.raises(ValueError, match="Invalid"):
        load_research_prompt("../etc/passwd")


def test_research_prompt_path():
    path = research_prompt_path("prompt_v2")
    assert path.name == "prompt_v2.md"
    assert path.parent == prompts_root()


def test_research_agent_uses_prompt_file(monkeypatch, tmp_path: Path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "prompt_v1.md").write_text(
        "CUSTOM_PROMPT_MARKER for research.\nUse tools carefully.",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "PROMPTS_DIR", str(prompt_dir))
    monkeypatch.setattr(settings, "PROMPT_VERSION", "prompt_v1")
    monkeypatch.setattr(settings, "SKILLS_DIR", str(tmp_path / "skills"))
    (tmp_path / "skills" / "verified_research").mkdir(parents=True)
    (tmp_path / "skills" / "verified_research" / "SKILL.md").write_text(
        "---\nname: verified_research\ndescription: test skill\n---\n\nBody.\n",
        encoding="utf-8",
    )

    class CaptureProvider:
        async def generate(self, messages, **kwargs):
            self.system = kwargs.get("system")
            return LLMResponse(content="done", tool_calls=[], stop_reason="end_turn")

        async def generate_structured(self, *args, **kwargs):
            raise NotImplementedError

    notes = EvidenceNotes(question="q")
    registry, clarification = build_research_registry(None, notes)
    provider = CaptureProvider()
    agent = ResearchAgent(
        provider,  # type: ignore[arg-type]
        registry,
        evidence_notes=notes,
        clarification=clarification,
        max_iterations=1,
        max_tool_calls=2,
    )
    import asyncio

    asyncio.run(agent.run("What is documented?"))
    assert "CUSTOM_PROMPT_MARKER" in (provider.system or "")
