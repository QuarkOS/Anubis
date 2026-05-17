import os
import re
import structlog
from google import genai
from google.genai import types

from anubis.domain.models import CompletionResult, Message, Role, ToolCall
from anubis.domain.protocols import LLMProvider
from anubis.domain.exceptions import LLMRateLimitError, LLMResponseError

logger = structlog.get_logger(__name__)

# Matches tags like <file:path/to/file.png>
FILE_TAG_PATTERN = re.compile(r"<file:(.+?)>")

class LLMGeminiProvider(LLMProvider):
    """Adapter for Google Gemini API providing multimodal generation capabilities."""

    def __init__(self, api_key: str):
        """Initialize the Gemini client using the provided API key."""
        if not api_key:
            logger.warning("gemini_provider_init", message="No API key provided.")
        self.client = genai.Client(api_key=api_key)

    async def generate_response(
        self,
        messages: list[Message],
        *,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: type | None = None,
    ) -> CompletionResult:
        """
        Execute a multimodal generation turn using the experimental premium Flex interactions tier,
        automatically uploading local file references.
        """
        system_instruction = None
        user_model_steps = []
        
        for i, msg in enumerate(messages):
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
                continue
                
            role = "user_input" if msg.role == Role.USER else "model_output"
            parts = []
            text_content = msg.content
            
            is_latest_message = (i == len(messages) - 1)
            file_matches = FILE_TAG_PATTERN.findall(text_content)
            
            if is_latest_message:
                for file_path in file_matches:
                    if os.path.isfile(file_path):
                        logger.info("gemini_uploading_file", file=file_path)
                        uploaded_file = self.client.files.upload(file=file_path)
                        mime = uploaded_file.mime_type or ""
                        if "audio" in mime:
                            parts.append({
                                "type": "audio",
                                "uri": uploaded_file.uri,
                                "mime_type": mime
                            })
                        elif "image" in mime:
                            parts.append({
                                "type": "image",
                                "uri": uploaded_file.uri,
                                "mime_type": mime
                            })
                        else:
                            parts.append({
                                "type": "document",
                                "uri": uploaded_file.uri,
                                "mime_type": mime
                            })
                    else:
                        logger.warning("gemini_file_missing_or_not_file", file=file_path)
            
            cleaned_text = FILE_TAG_PATTERN.sub("", text_content).strip()
            if cleaned_text:
                parts.append({"type": "text", "text": cleaned_text})
                
            if parts:
                user_model_steps.append({
                    "type": role,
                    "content": parts
                })
                
        if not user_model_steps:
            user_model_steps = [{"type": "user_input", "content": [{"type": "text", "text": "Hello"}]}]

        response_format_arg = None
        if response_format:
            response_format_arg = {
                "type": "text",
                "mime_type": "application/json",
                "schema": response_format.model_json_schema()
            }
            
        response_mime_type = "application/json" if response_format else "text/plain"
        tools_arg = [{"type": "google_search"}]
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        try:
            interaction = self.client.interactions.create(
                model=model,
                input=user_model_steps,
                service_tier='flex',
                generation_config=generation_config,
                system_instruction=system_instruction,
                response_format=response_format_arg,
                response_mime_type=response_mime_type,
                tools=tools_arg,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                raise LLMRateLimitError("Gemini Rate Limit Hit", retry_after=5.0) from e
            raise LLMResponseError(f"Gemini API Error: {e}") from e

        full_text = ""
        for step in reversed(interaction.steps):
            if step.type == "model_output" and hasattr(step, "content") and step.content:
                full_text = "".join(
                    part.text for part in step.content if hasattr(part, "text") and part.text
                )
                if full_text:
                    break
        if not full_text:
            full_text = interaction.steps[-1].content[0].text if (interaction.steps and interaction.steps[-1].content) else ""
        
        logger.info("gemini_response", status=interaction.status, text_length=len(full_text))

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(interaction, "usage") and interaction.usage:
            prompt_tokens = getattr(interaction.usage, "total_input_tokens", 0) or 0
            completion_tokens = getattr(interaction.usage, "total_output_tokens", 0) or 0

        return CompletionResult(
            content=full_text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def estimate_token_count(self, text: str, *, model: str = "gemini-3-flash-preview") -> int:
        """Provide a simple character-based heuristic for token estimation."""
        return len(text) // 4
