"""I/O boundary schemas — API request/response contracts.

These schemas are the Pydantic models that sit at the edges of the system:
FastAPI request bodies, serialized API responses, and validated LLM
structured-output payloads.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Schema for incoming chat requests."""

    conversation_id: UUID | None = None
    message: str = Field(..., min_length=1, max_length=32_000)
    model: str = "default"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128_000)


class ChatResponse(BaseModel):
    """Schema for outbound chat responses."""

    conversation_id: UUID
    reply: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class ErrorResponse(BaseModel):
    """Generic error message envelope."""

    error: str
    detail: str | None = None
    retry_after: float | None = None


class ExtractedEntity(BaseModel):
    """Single data entity identified by the LLM."""

    name: str
    entity_type: str
    confidence: float = Field(ge=0.0, le=1.0)


class StructuredExtractionResult(BaseModel):
    """Result container for structured entity extraction tasks."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    summary: str = ""
