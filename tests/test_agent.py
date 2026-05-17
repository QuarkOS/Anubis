"""Tests for the AgentService."""

from __future__ import annotations

from uuid import uuid4

import pytest

from anubis.domain.models import CompletionResult, Conversation, Message, Role
from anubis.domain.schemas import ChatRequest
from anubis.prompts import PromptRegistry
from anubis.repositories.conversation_memory import InMemoryConversationRepository
from anubis.services.agent import AgentService
from anubis.services.context import SlidingWindowContextBuilder


class FakeLLM:
    """Minimal LLM stub for unit tests."""

    def __init__(self, response: str = "Hello from the LLM") -> None:
        self._response = response
        self.last_messages: list[Message] = []

    async def generate_response(
        self,
        messages: list[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type | None = None,
    ) -> CompletionResult:
        self.last_messages = messages
        return CompletionResult(
            content=self._response,
            prompt_tokens=10,
            completion_tokens=5,
            model="fake-model",
        )

    async def estimate_token_count(self, text: str, *, model: str = "default") -> int:
        return len(text.split())


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def agent(fake_llm: FakeLLM) -> AgentService:
    repo = InMemoryConversationRepository()
    ctx = SlidingWindowContextBuilder(llm=fake_llm)
    prompts = PromptRegistry()
    return AgentService(llm=fake_llm, repo=repo, context_builder=ctx, prompts=prompts)


@pytest.mark.asyncio
async def test_chat_returns_valid_response(agent: AgentService) -> None:
    req = ChatRequest(message="What is Clean Architecture?")
    resp = await agent.chat(req)
    assert resp.reply == "Hello from the LLM"
    assert resp.model == "fake-model"


@pytest.mark.asyncio
async def test_chat_persists_conversation(agent: AgentService) -> None:
    req = ChatRequest(message="First message")
    resp = await agent.chat(req)
    # Second turn should reuse conversation
    req2 = ChatRequest(conversation_id=resp.conversation_id, message="Second message")
    resp2 = await agent.chat(req2)
    assert resp2.conversation_id == resp.conversation_id
