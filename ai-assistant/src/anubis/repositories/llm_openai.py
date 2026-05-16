"""OpenAI-compatible LLM repository.

Concrete implementation of the ``LLMProvider`` protocol using the
OpenAI-compatible chat completions API via ``httpx``.  Works with
OpenAI, Azure OpenAI, OpenRouter, and any compatible endpoint.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
import tiktoken

from anubis.domain.exceptions import LLMRateLimitError, LLMResponseError
from anubis.domain.models import CompletionResult, Message

logger = structlog.get_logger(__name__)


class OpenAILLMProvider:
    """LLM provider targeting OpenAI-compatible APIs using httpx."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout),
        )

    async def generate_response(
        self,
        messages: list[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type | None = None,
    ) -> CompletionResult:
        """Send a chat completion request and return the LLM response."""
        resolved_model = model if model != "default" else self._default_model

        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMResponseError(f"Request timed out: {exc}") from exc

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("retry-after", "1"))
            raise LLMRateLimitError(retry_after=retry_after)

        if resp.status_code >= 400:
            raise LLMResponseError(
                f"LLM API error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return CompletionResult(
            content=choice["message"].get("content"),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", resolved_model),
        )

    async def estimate_token_count(self, text: str, *, model: str = "default") -> int:
        """Calculate the number of tokens in the text using tiktoken."""
        resolved = model if model != "default" else self._default_model
        try:
            enc = tiktoken.encoding_for_model(resolved)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
