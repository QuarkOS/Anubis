"""In-memory conversation repository.

A simple reference implementation of ``ConversationRepository`` suitable
for development and testing.  Replace with a database-backed implementation
(e.g., PostgreSQL + asyncpg, Redis, SQLite) for production use.
"""

from __future__ import annotations

from uuid import UUID

from anubis.domain.models import Conversation


class InMemoryConversationRepository:
    """Thread-safe in-memory store for conversation history."""

    def __init__(self) -> None:
        self._store: dict[UUID, Conversation] = {}

    async def fetch_conversation(self, conversation_id: UUID) -> Conversation | None:
        """Retrieve a conversation from memory by ID."""
        return self._store.get(conversation_id)

    async def persist_conversation(self, conversation: Conversation) -> None:
        """Store or update a conversation in memory."""
        self._store[conversation.id] = conversation

    async def remove_conversation(self, conversation_id: UUID) -> None:
        """Remove a conversation from memory."""
        self._store.pop(conversation_id, None)
