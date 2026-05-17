"""Comprehensive unit tests covering uncovered edge cases across the Anubis assistant codebase.

This includes edge cases in:
- SlidingWindowContextBuilder (very small token budgets, empty histories)
- AgentService (JSON parser robustness, invalid UUID strings, rate limit retry backoff)
- LLMGeminiProvider (token counts from usage metadata, directories/missing files uploader filter)
"""

from __future__ import annotations

import json
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from anubis.domain.exceptions import LLMJsonParseError, LLMRateLimitError, LLMResponseError
from anubis.domain.models import CompletionResult, Conversation, Message, Role, ToolCall
from anubis.domain.schemas import ChatRequest, StructuredExtractionResult
from anubis.prompts import PromptRegistry
from anubis.repositories.conversation_memory import InMemoryConversationRepository
from anubis.repositories.llm_gemini import LLMGeminiProvider
from anubis.services.agent import AgentService
from anubis.services.context import SlidingWindowContextBuilder


class MockLLMForContext:
    """Mock LLM to provide controlled token counts for context builder tests."""

    async def estimate_token_count(self, text: str, *, model: str = "default") -> int:
        # Simple rule: each word is 1 token
        return len(text.split())

    async def generate_response(
        self,
        messages: list[Message],
        *,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type | None = None,
    ) -> CompletionResult:
        return CompletionResult(content="Hello", prompt_tokens=1, completion_tokens=1)


@pytest.mark.asyncio
async def test_context_builder_extremely_small_budget() -> None:
    """Edge Case: The token budget is smaller than the system prompt + latest message.

    It should return [system_msg, latest] to guarantee essential context even if it overflows.
    """
    llm = MockLLMForContext()
    builder = SlidingWindowContextBuilder(llm=llm)

    system_prompt = "You are a helpful assistant."  # 5 tokens
    conversation = Conversation()
    conversation.messages = [
        Message(role=Role.USER, content="Hello"),  # 1 token
        Message(role=Role.ASSISTANT, content="Hi there"),  # 2 tokens
        Message(role=Role.USER, content="Explain Clean Architecture please"),  # 4 tokens
    ]

    # system (5) + latest (4) = 9 tokens. Let's set max_context_tokens to 7 (budget < 9)
    result = await builder.build_message_context(
        conversation,
        system_prompt=system_prompt,
        max_context_tokens=7,
    )

    # Should only return system message and latest message
    assert len(result) == 2
    assert result[0].role == Role.SYSTEM
    assert result[1].content == "Explain Clean Architecture please"


@pytest.mark.asyncio
async def test_context_builder_empty_history() -> None:
    """Edge Case: The conversation has no messages at all."""
    llm = MockLLMForContext()
    builder = SlidingWindowContextBuilder(llm=llm)

    conversation = Conversation()
    result = await builder.build_message_context(
        conversation,
        system_prompt="System prompt",
        max_context_tokens=100,
    )

    # Should only return the system message
    assert len(result) == 1
    assert result[0].role == Role.SYSTEM
    assert result[0].content == "System prompt"


@pytest.mark.asyncio
async def test_agent_invalid_uuid_format() -> None:
    """Edge Case: Incoming request has an invalid/malformed conversation_id string.

    The AgentService should handle the ValueError gracefully and create a new session.
    """
    llm = MockLLMForContext()
    repo = InMemoryConversationRepository()
    ctx = SlidingWindowContextBuilder(llm=llm)
    prompts = PromptRegistry()
    agent = AgentService(llm=llm, repo=repo, context_builder=ctx, prompts=prompts)

    # Call the private helper directly to verify ValueError handling
    conv = await agent._load_or_create_conversation("invalid-uuid-string")
    assert conv.id is not None


def test_agent_json_parsing_robustness() -> None:
    """Edge Case: LLM returns JSON with markdown fences, extra commentary, or nested fences."""
    # 1. JSON wrapped in ```json ... ``` codeblock
    raw_output_1 = """
    ```json
    {
      "entities": [{"name": "Anubis", "entity_type": "Product", "confidence": 0.95}],
      "summary": "This is Anubis."
    }
    ```
    """
    parsed_1 = AgentService._parse_llm_json_output(raw_output_1, StructuredExtractionResult)
    assert len(parsed_1.entities) == 1
    assert parsed_1.entities[0].name == "Anubis"
    assert parsed_1.summary == "This is Anubis."

    # 2. JSON wrapped in ``` ... ``` codeblock without language specifier
    raw_output_2 = """
    ```
    {
      "entities": [],
      "summary": "Empty"
    }
    ```
    """
    parsed_2 = AgentService._parse_llm_json_output(raw_output_2, StructuredExtractionResult)
    assert len(parsed_2.entities) == 0
    assert parsed_2.summary == "Empty"

    # 3. JSON with surrounding conversational commentary
    raw_output_3 = """
    Here is the requested extraction result:
    {
      "entities": [{"name": "Emilio", "entity_type": "Person", "confidence": 1.0}],
      "summary": "User Emilio detected."
    }
    Hope this helps!
    """
    parsed_3 = AgentService._parse_llm_json_output(raw_output_3, StructuredExtractionResult)
    assert parsed_3.entities[0].name == "Emilio"

    # 4. Invalid JSON structure should raise LLMJsonParseError
    with pytest.raises(LLMJsonParseError):
        AgentService._parse_llm_json_output("{invalid json}", StructuredExtractionResult)


@pytest.mark.asyncio
async def test_agent_rate_limit_backoff_exhaustion() -> None:
    """Edge Case: LLM continuously returns RateLimitErrors.

    The AgentService should attempt backoff and eventually raise LLMRateLimitError.
    """
    class RateLimitedLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_response(self, *args, **kwargs):
            self.calls += 1
            raise LLMRateLimitError("Too Many Requests", retry_after=0.01)

        async def estimate_token_count(self, text: str, *, model: str = "default") -> int:
            return 1

    llm = RateLimitedLLM()
    repo = InMemoryConversationRepository()
    ctx = SlidingWindowContextBuilder(llm=llm)  # type: ignore
    prompts = PromptRegistry()
    agent = AgentService(llm=llm, repo=repo, context_builder=ctx, prompts=prompts)  # type: ignore

    messages = [Message(role=Role.USER, content="Hello")]

    with pytest.raises(LLMRateLimitError):
        await agent._request_completion_with_retry(messages)

    # Should attempt exactly 3 times before giving up
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_gemini_provider_token_count_usage_metadata() -> None:
    """Edge Case: Gemini provider extracts real token counts from usage_metadata when available."""
    api_key = "fake_gemini_key"
    
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        provider = LLMGeminiProvider(api_key=api_key)
        
        # Mock API Response
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"
        mock_candidate.content.parts = [MagicMock(text="Gemini Answer")]
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.text = "Gemini Answer"
        
        # Populate real usage metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 105
        mock_usage.candidates_token_count = 42
        mock_response.usage_metadata = mock_usage
        
        mock_client.models.generate_content.return_value = mock_response
        
        messages = [Message(role=Role.USER, content="Tell me a joke.")]
        result = await provider.generate_response(messages)
        
        # Verify correct metadata parsing
        assert result.content == "Gemini Answer"
        assert result.prompt_tokens == 105
        assert result.completion_tokens == 42


@pytest.mark.asyncio
async def test_gemini_provider_file_uploader_filter() -> None:
    """Edge Case: A user prompt references a missing file, a directory, or has blank text.

    LLMGeminiProvider should filter out non-files (folders, nonexistent paths) to prevent uploader crashes.
    """
    api_key = "fake_gemini_key"
    
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        provider = LLMGeminiProvider(api_key=api_key)
        
        # Mock file system: 'valid.png' is a file, 'src/anubis' is a directory, 'missing.txt' doesn't exist
        def mock_isfile(path: str) -> bool:
            return path == "valid.png"
            
        with patch("os.path.isfile", side_effect=mock_isfile):
            # Mock files.upload to return a valid result with string attributes to satisfy Pydantic validation
            mock_uploaded_file = MagicMock()
            mock_uploaded_file.uri = "https://example.com/file"
            mock_uploaded_file.mime_type = "image/png"
            mock_client.files.upload.return_value = mock_uploaded_file

            # Mock generate_content
            mock_client.models.generate_content.return_value = MagicMock(
                candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text="Processed")]))],
                text="Processed",
                usage_metadata=None
            )
            
            # Message referencing a real file, a directory, and a missing path
            messages = [
                Message(
                    role=Role.USER, 
                    content="Check this: <file:valid.png> and <file:src/anubis> and <file:missing.txt>"
                )
            ]
            
            await provider.generate_response(messages)
            
            # Verify client.files.upload was only called ONCE (for the valid file)
            mock_client.files.upload.assert_called_once_with(file="valid.png")
