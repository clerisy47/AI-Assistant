"""The main agentic assistant endpoint.

Wires together everything: pulls the configured (or per-request-overridden)
LLM provider, builds a tool registry (calculator, datetime, knowledge-base
search), and runs the multi-turn tool-calling loop via AgentOrchestrator.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agent.orchestrator import AgentOrchestrator
from app.config import settings
from app.dependencies import get_retriever
from app.llm.base import LLMMessage
from app.llm.factory import get_llm_provider
from app.rag.retriever import Retriever
from app.schemas.chat import ChatRequest, ChatResponse
from app.tools.builtin_tools import CALCULATOR_DEFINITION, DATETIME_DEFINITION, calculator, get_current_datetime
from app.tools.knowledge_base_tool import KNOWLEDGE_BASE_DEFINITION, make_knowledge_base_tool
from app.tools.registry import ToolRegistry

router = APIRouter(tags=["chat"])


def _build_tool_registry(retriever: Retriever) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CALCULATOR_DEFINITION, calculator)
    registry.register(DATETIME_DEFINITION, get_current_datetime)
    registry.register(KNOWLEDGE_BASE_DEFINITION, make_knowledge_base_tool(retriever))
    return registry


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, retriever: Retriever = Depends(get_retriever)) -> ChatResponse:
    provider = get_llm_provider(payload.provider)
    registry = _build_tool_registry(retriever)
    orchestrator = AgentOrchestrator(provider, registry, max_tool_iterations=settings.MAX_TOOL_ITERATIONS)

    history = [LLMMessage(role=m.role, content=m.content) for m in payload.messages]
    result = await orchestrator.run(history, temperature=payload.temperature, top_p=payload.top_p, max_tokens=settings.MAX_TOKENS)

    return ChatResponse(**result)
