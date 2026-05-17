"""Protocol definitions for dependency injection.

All abstractions that services depend on are defined here as Protocols.
Concrete implementations in the repositories layer satisfy these contracts.
This keeps the service layer decoupled from any specific provider or database.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from anubis.domain.models import CompletionResult, Conversation, Message
from anubis.domain.schemas import SystemState


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM backends providing multimodal generation and tokenization utilities."""

    async def generate_response(
        self,
        messages: list[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type | None = None,
    ) -> CompletionResult:
        """Execute a generation turn, supporting text, audio, and vision parts."""
        ...

    async def estimate_token_count(self, text: str, *, model: str = "default") -> int:
        """Provide a heuristic or model-specific estimate of the token count for a string."""
        ...


@runtime_checkable
class ConversationRepository(Protocol):
    """Interface for persisting and retrieving stateful conversation history."""

    async def fetch_conversation(self, conversation_id: UUID) -> Conversation | None:
        """Retrieve a specific conversation and its message history by ID."""
        ...

    async def persist_conversation(self, conversation: Conversation) -> None:
        """Commit a conversation to the storage layer, updating existing entries."""
        ...

    async def remove_conversation(self, conversation_id: UUID) -> None:
        """Permanently delete a conversation and its history."""
        ...


@runtime_checkable
class ContextBuilder(Protocol):
    """Interface for assembling and budgeting the message context for LLM turns."""

    async def build_message_context(
        self,
        conversation: Conversation,
        *,
        system_prompt: str,
        max_context_tokens: int,
    ) -> list[Message]:
        """Assemble a prioritized list of messages that fits within the token budget."""
        ...


@runtime_checkable
class SystemProbe(Protocol):
    """Interface for gathering real-time system telemetry and user context."""

    async def probe_state(self) -> SystemState:
        """Capture a snapshot of the current hardware resources and ambient environment."""
        ...
