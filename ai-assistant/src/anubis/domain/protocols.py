"""Protocol definitions for dependency injection.

All abstractions that services depend on are defined here as Protocols.
Concrete implementations in the repositories layer satisfy these contracts.
This keeps the service layer decoupled from any specific provider or database.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from anubis.domain.models import CompletionResult, Conversation, Message


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM backends providing generation and tokenization utilities."""

    async def generate_response(
        self,
        messages: list[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type | None = None,
    ) -> CompletionResult:
        """Send messages to the LLM and return a completion result."""
        ...

    async def estimate_token_count(self, text: str, *, model: str = "default") -> int:
        """Estimate the number of tokens in the given text."""
        ...


@runtime_checkable
class ConversationRepository(Protocol):
    """Interface for persisting and retrieving conversation history."""

    async def fetch_conversation(self, conversation_id: UUID) -> Conversation | None:
        """Retrieve a conversation by its unique identifier."""
        ...

    async def persist_conversation(self, conversation: Conversation) -> None:
        """Save or update a conversation in the storage layer."""
        ...

    async def remove_conversation(self, conversation_id: UUID) -> None:
        """Delete a conversation from the storage layer."""
        ...


@runtime_checkable
class ContextBuilder(Protocol):
    """Interface for assembling the message context window for LLM requests."""

    async def build_message_context(
        self,
        conversation: Conversation,
        *,
        system_prompt: str,
        max_context_tokens: int,
    ) -> list[Message]:
        """Produce a token-budgeted list of messages ready for the LLM."""
        ...
