"""Verified-research endpoint: supervisor coordinates Research ↔ Verifier.

Keeps POST /chat unchanged. MVP uses the last user message as the question.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.agent.supervisor import ResearchSupervisor
from app.dependencies import get_retriever
from app.llm.factory import get_llm_provider
from app.rag.retriever import Retriever
from app.schemas.research import ResearchRequest, ResearchResponse

router = APIRouter(tags=["research"])


def _last_user_question(payload: ResearchRequest) -> str:
    for message in reversed(payload.messages):
        if message.role == "user" and (message.content or "").strip():
            return message.content.strip()
    raise HTTPException(status_code=422, detail="At least one non-empty user message is required.")


@router.post("/research", response_model=ResearchResponse)
async def research(
    payload: ResearchRequest,
    retriever: Optional[Retriever] = Depends(get_retriever),
) -> ResearchResponse:
    question = _last_user_question(payload)
    provider = get_llm_provider(payload.provider)
    supervisor = ResearchSupervisor.create(
        provider,
        retriever=retriever,
        max_iterations=payload.max_iterations,
    )
    result = await supervisor.run(
        question,
        temperature=payload.temperature,
        top_p=payload.top_p,
    )
    return ResearchResponse(**result)
