"""FastAPI application initialization and dependency injection wiring."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
import uvicorn
from fastapi import FastAPI

from anubis.config import get_settings
from anubis.interfaces.routes import router
from anubis.prompts import get_prompt_registry
from anubis.repositories.conversation_memory import InMemoryConversationRepository
from anubis.repositories.llm_openai import OpenAILLMProvider
from anubis.services.agent import AgentService
from anubis.services.context import SlidingWindowContextBuilder

logger = structlog.get_logger(__name__)
_llm_provider: OpenAILLMProvider | None = None
_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    """Return the initialized AgentService singleton."""
    if _agent_service is None:
        raise RuntimeError("AgentService not initialized")
    return _agent_service


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage application-level resources and service initialization."""
    global _llm_provider, _agent_service  # noqa: PLW0603
    settings = get_settings()
    _llm_provider = OpenAILLMProvider(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        default_model=settings.llm_default_model,
        timeout=settings.llm_timeout,
    )
    repo = InMemoryConversationRepository()
    ctx = SlidingWindowContextBuilder(llm=_llm_provider)
    _agent_service = AgentService(
        llm=_llm_provider, repo=repo, context_builder=ctx, prompts=get_prompt_registry()
    )
    logger.info("started", model=settings.llm_default_model)
    yield
    await _llm_provider.close()


def create_app() -> FastAPI:
    """Factory to create and configure the FastAPI application."""
    s = get_settings()
    app = FastAPI(title=s.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(router, prefix="/api/v1")
    return app


def main() -> None:
    s = get_settings()
    uvicorn.run(
        "anubis.interfaces.api:create_app",
        factory=True, host=s.host, port=s.port,
        reload=s.debug, log_level=s.log_level.lower(),
    )

if __name__ == "__main__":
    main()
