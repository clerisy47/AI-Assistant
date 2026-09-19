"""Unit tests for INJECT_FAILURE on search_knowledge_base (Phase 7)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.knowledge_base_tool import (
    KNOWLEDGE_BASE_DEFINITION,
    KnowledgeBaseInjectedFailure,
    make_knowledge_base_tool,
    resolve_inject_failure,
)
from app.tools.registry import ToolRegistry


@pytest.mark.parametrize(
    "mode,marker",
    [
        ("kb_unavailable", "knowledge base unavailable"),
        ("kb_timeout", "timed out"),
        ("kb_malformed", "malformed"),
    ],
)
def test_resolve_inject_failure_raises_with_marker(mode: str, marker: str):
    with pytest.raises(KnowledgeBaseInjectedFailure) as exc_info:
        resolve_inject_failure(mode)
    assert marker in str(exc_info.value).lower()


def test_resolve_inject_failure_empty_is_noop():
    resolve_inject_failure("")
    resolve_inject_failure("   ")


def test_resolve_inject_failure_unknown_mode_is_noop():
    resolve_inject_failure("not_a_real_mode")


@pytest.mark.parametrize(
    "mode,marker",
    [
        ("kb_unavailable", "knowledge base unavailable"),
        ("kb_timeout", "timed out"),
        ("kb_malformed", "malformed"),
    ],
)
def test_kb_tool_inject_never_calls_retriever(mode: str, marker: str):
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])
    handler = make_knowledge_base_tool(retriever)

    with patch("app.tools.knowledge_base_tool.settings") as mock_settings:
        mock_settings.INJECT_FAILURE = mode
        with pytest.raises(KnowledgeBaseInjectedFailure) as exc_info:
            asyncio.run(handler(query="anything", top_k=4))
        assert marker in str(exc_info.value).lower()
    retriever.retrieve.assert_not_called()


@pytest.mark.parametrize(
    "mode,marker",
    [
        ("kb_unavailable", "knowledge base unavailable"),
        ("kb_timeout", "timed out"),
        ("kb_malformed", "malformed"),
    ],
)
def test_registry_surfaces_inject_as_is_error(mode: str, marker: str):
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])
    registry = ToolRegistry()
    registry.register(KNOWLEDGE_BASE_DEFINITION, make_knowledge_base_tool(retriever))

    with patch("app.tools.knowledge_base_tool.settings") as mock_settings:
        mock_settings.INJECT_FAILURE = mode
        result, is_error = asyncio.run(
            registry.call("search_knowledge_base", {"query": "q", "top_k": 4})
        )
    assert is_error is True
    assert marker in result.lower()
    retriever.retrieve.assert_not_called()
