"""Tests the `/health` handler function directly rather than through
TestClient + the full app lifespan, so this test needs neither a live
Qdrant nor an LLM API key -- it only exercises config resolution."""

import asyncio

from app.api.health import health
from app.config import settings


class _FakeState:
    def __init__(self, ready: bool):
        self.vector_store_ready = ready


class _FakeApp:
    def __init__(self, ready: bool):
        self.state = _FakeState(ready)


class _FakeRequest:
    def __init__(self, ready: bool):
        self.app = _FakeApp(ready)


def test_health_reports_ok_status_and_configured_provider():
    result = asyncio.run(health(_FakeRequest(ready=True)))
    assert result["status"] == "ok"
    assert result["llm_provider"] == settings.LLM_PROVIDER
    assert result["vector_store_ready"] is True


def test_health_reflects_vector_store_not_ready():
    result = asyncio.run(health(_FakeRequest(ready=False)))
    assert result["vector_store_ready"] is False


def test_health_active_model_matches_configured_provider():
    result = asyncio.run(health(_FakeRequest(ready=True)))
    expected = {
        "anthropic": settings.ANTHROPIC_MODEL,
        "openai": settings.OPENAI_MODEL,
        "local": settings.VLLM_MODEL,
    }[settings.LLM_PROVIDER.lower()]
    assert result["active_model"] == expected
