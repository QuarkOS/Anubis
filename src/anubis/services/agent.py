"""AgentService — core orchestrator for the AI assistant.

This is the central service that wires together:
  • An LLM provider (injected via Protocol)
  • A conversation repository (injected via Protocol)
  • A context builder (injected via Protocol)
  • A prompt registry (configuration, not a Protocol)

It validates all I/O through Pydantic schemas and handles LLM-specific
failure modes (rate limits, hallucinated JSON) gracefully.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import structlog
from pydantic import ValidationError

from anubis.domain.exceptions import LLMJsonParseError, LLMRateLimitError
from anubis.domain.models import CompletionResult, Conversation, Message, Role
from anubis.domain.protocols import ContextBuilder, ConversationRepository, LLMProvider
from anubis.domain.schemas import (
    ChatRequest,
    ChatResponse,
    StructuredExtractionResult,
)
from anubis.prompts import PromptRegistry

logger = structlog.get_logger(__name__)

_MAX_JSON_RETRIES = 2
_MAX_CONTEXT_TOKENS = 120_000


class AgentService:
    """Orchestrates LLM interactions, memory management, and context building."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        repo: ConversationRepository,
        context_builder: ContextBuilder,
        prompts: PromptRegistry,
    ) -> None:
        self._llm = llm
        self._repo = repo
        self._ctx = context_builder
        self._prompts = prompts

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat turn by building context, calling the LLM, and persisting the interaction."""
        log = logger.bind(conversation_id=str(request.conversation_id))
        conversation = await self._load_or_create_conversation(request.conversation_id)

        user_msg = Message(role=Role.USER, content=request.message)
        conversation.messages.append(user_msg)

        system_prompt = self._prompts.render_prompt(
            "system.default",
            current_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )
        context = await self._ctx.build_message_context(
            conversation,
            system_prompt=system_prompt,
            max_context_tokens=_MAX_CONTEXT_TOKENS,
        )

        result = await self._request_completion_with_retry(
            context,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=result.content or "",
            token_count=result.completion_tokens,
        )
        conversation.messages.append(assistant_msg)
        await self._repo.persist_conversation(conversation)

        log.info(
            "chat_turn_complete",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

        return ChatResponse(
            conversation_id=conversation.id,
            reply=assistant_msg.content,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    async def extract_structured(
        self, text: str, *, model: str = "default"
    ) -> StructuredExtractionResult:
        """Extract structured entities from text with automatic retry on JSON parsing failures."""
        system_prompt = self._prompts.render_prompt("system.extraction")
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=text),
        ]

        last_error: LLMJsonParseError | None = None
        for attempt in range(_MAX_JSON_RETRIES + 1):
            result = await self._request_completion_with_retry(messages, model=model)
            raw = (result.content or "").strip()

            try:
                parsed = self._parse_llm_json_output(raw, StructuredExtractionResult)
                return parsed
            except LLMJsonParseError as exc:
                last_error = exc
                logger.warning(
                    "llm_json_parse_failed",
                    attempt=attempt + 1,
                    raw_output=raw[:200],
                )
                messages.append(Message(role=Role.ASSISTANT, content=raw))
                messages.append(
                    Message(
                        role=Role.USER,
                        content=(
                            "Your previous response was not valid JSON. "
                            "Please respond ONLY with the JSON object, no markdown fences."
                        ),
                    )
                )

        raise last_error  # type: ignore[misc]

    async def _load_or_create_conversation(self, conversation_id: str | None) -> Conversation:
        """Retrieve a conversation by ID or initialize a new one."""
        if conversation_id:
            from uuid import UUID

            existing = await self._repo.fetch_conversation(UUID(str(conversation_id)))
            if existing:
                return existing
        return Conversation(id=uuid4())

    async def _request_completion_with_retry(
        self,
        messages: list[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Call the LLM and retry on rate limits using exponential backoff."""
        import asyncio

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                return await self._llm.generate_response(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except LLMRateLimitError as exc:
                if attempt == max_attempts - 1:
                    raise
                wait = exc.retry_after * (2**attempt)
                logger.warning("rate_limited", retry_after=wait, attempt=attempt + 1)
                await asyncio.sleep(wait)

        msg = "Exhausted LLM retry attempts"
        raise RuntimeError(msg)

    @staticmethod
    def _parse_llm_json_output(raw: str, schema: type[StructuredExtractionResult]) -> StructuredExtractionResult:
        """Parse and validate JSON from raw LLM output, handling common formatting hallucinations."""
        cleaned = raw
        if "```" in cleaned:
            lines = cleaned.split("\n")
            json_lines: list[str] = []
            inside = False
            for line in lines:
                if line.strip().startswith("```"):
                    inside = not inside
                    continue
                if inside:
                    json_lines.append(line)
            if json_lines:
                cleaned = "\n".join(json_lines)

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMJsonParseError(
                f"Failed to parse JSON: {exc}",
                raw_output=raw,
            ) from exc

        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMJsonParseError(
                f"JSON schema validation failed: {exc}",
                raw_output=raw,
            ) from exc
