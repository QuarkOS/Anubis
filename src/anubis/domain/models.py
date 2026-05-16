"""Core domain models and value objects representing the fundamental data structures of the assistant."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Participant role in a conversation turn."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Single message within a conversation."""

    role: Role
    content: str
    name: str | None = None
    token_count: int | None = None


class Conversation(BaseModel):
    """Ordered sequence of messages forming a dialogue."""

    id: UUID = Field(default_factory=uuid4)
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_tokens(self) -> int:
        """Sum of all estimated token counts in the conversation."""
        return sum(m.token_count or 0 for m in self.messages)


class ToolCall(BaseModel):
    """Structured representation of an LLM-requested tool invocation."""

    id: str
    name: str
    arguments: dict[str, object]


class CompletionResult(BaseModel):
    """Normalized response from any LLM provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
