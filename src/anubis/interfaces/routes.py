"""API route handlers for chat and extraction services."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from anubis.domain.exceptions import LLMRateLimitError, LLMResponseError
from anubis.domain.schemas import ChatRequest, ChatResponse, ErrorResponse

router = APIRouter(tags=["chat"])
logger = structlog.get_logger(__name__)


@router.post("/chat", response_model=ChatResponse, responses={429: {"model": ErrorResponse}})
async def chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests by delegating to the AgentService."""
    from anubis.interfaces.api import get_agent_service
    svc = get_agent_service()
    try:
        return await svc.chat(request)
    except LLMRateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=ErrorResponse(
                error="rate_limited", retry_after=exc.retry_after
            ).model_dump(),
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/extract")
async def extract(text: str) -> dict[str, object]:
    """Perform structured extraction from the provided text."""
    from anubis.interfaces.api import get_agent_service
    svc = get_agent_service()
    result = await svc.extract_structured(text)
    return result.model_dump()


@router.get("/health")
async def health() -> dict[str, str]:
    """Return the health status of the API."""
    return {"status": "ok"}
