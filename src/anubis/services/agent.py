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
from anubis.domain.protocols import ContextBuilder, ConversationRepository, LLMProvider, SystemProbe
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
    """Orchestrates LLM interactions, memory management, and situational context building."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        repo: ConversationRepository,
        context_builder: ContextBuilder,
        prompts: PromptRegistry,
        system_probe: SystemProbe | None = None,
    ) -> None:
        """Initialize the service with its primary domain dependencies."""
        self._llm = llm
        self._repo = repo
        self._ctx = context_builder
        self._prompts = prompts
        self._system = system_probe

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Execute a complete chat turn, including situational telemetry and history management.
        
        Captures the current system state, renders the system prompt, assembles 
        the context window, and executes the LLM completion.
        """
        log = logger.bind(conversation_id=str(request.conversation_id))
        conversation = await self._load_or_create_conversation(request.conversation_id)

        user_msg = Message(role=Role.USER, content=request.message)
        conversation.messages.append(user_msg)

        system_state = None
        if self._system:
            system_state = await self._system.probe_state()

        system_prompt = self._prompts.render_prompt(
            "system.default",
            current_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            system_state=system_state,
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
        """
        Extract typed entities from raw text using a structured JSON schema.
        
        Includes automatic retry logic to handle common LLM JSON formatting hallucinations.
        """
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
        """Retrieve a conversation by ID or initialize a new one if missing or invalid."""
        if conversation_id:
            from uuid import UUID
            try:
                existing = await self._repo.fetch_conversation(UUID(str(conversation_id)))
                if existing:
                    return existing
            except ValueError:
                logger.warning("invalid_conversation_uuid", conversation_id=conversation_id)
        return Conversation(id=uuid4())

    async def _request_completion_with_retry(
        self,
        messages: list[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Execute an LLM request with exponential backoff on rate limit errors."""
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
        """
        Clean and validate raw LLM output against a Pydantic schema.
        
        Surgically extracts JSON objects from markdown fences and handles common truncation errors.
        """
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
