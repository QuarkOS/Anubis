"""Domain-level exceptions.

These are raised by repositories and caught/handled by services and interfaces.
Keeping exceptions in the domain layer allows services to reference them
without importing repository internals.
"""

from __future__ import annotations


class AnubisError(Exception):
    """Base exception for all errors within the Anubis assistant."""


class LLMError(AnubisError):
    """Base exception for LLM provider failures."""


class LLMRateLimitError(LLMError):
    """Exception raised when the LLM provider returns a rate limit error."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMResponseError(LLMError):
    """Exception for non-retryable LLM provider errors."""


class LLMJsonParseError(LLMError):
    """Exception raised when LLM output cannot be parsed as valid JSON."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class RepositoryError(AnubisError):
    """Base exception for failures in the persistence layer."""


class EntityNotFoundError(RepositoryError):
    """Exception raised when a requested database entity is missing."""
