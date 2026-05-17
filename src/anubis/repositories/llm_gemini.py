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
        Execute a multimodal generation turn, automatically uploading local file references.
        
        Parses the latest user message for <file:path> tags, uploads identified 
        media to the Gemini File API, and injects them as active context parts.
        """
        
        gemini_contents = []
        for i, msg in enumerate(messages):
            role = "user" if msg.role in (Role.USER, Role.SYSTEM) else "model"
            
            parts = []
            text_content = msg.content
            
            is_latest_message = (i == len(messages) - 1)
            file_matches = FILE_TAG_PATTERN.findall(text_content)
            
            if is_latest_message:
                for file_path in file_matches:
                    if os.path.isfile(file_path):
                        logger.info("gemini_uploading_file", file=file_path)
                        uploaded_file = self.client.files.upload(file=file_path)
                        file_part = types.Part(
                            file_data=types.FileData(
                                file_uri=uploaded_file.uri, 
                                mime_type=uploaded_file.mime_type
                            )
                        )
                        parts.append(file_part)
                    else:
                        logger.warning("gemini_file_missing_or_not_file", file=file_path)
            
            cleaned_text = FILE_TAG_PATTERN.sub("", text_content).strip()
            if cleaned_text:
                parts.append(types.Part(text=cleaned_text))
                
            if not parts:
                continue

            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=parts
                )
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if response_format else "text/plain",
            response_schema=response_format.model_json_schema() if response_format else None,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=gemini_contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str:
                raise LLMRateLimitError("Gemini Rate Limit Hit", retry_after=5.0) from e
            raise LLMResponseError(f"Gemini API Error: {e}") from e

        full_text = ""
        finish_reason = "unknown"
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = str(getattr(candidate, 'finish_reason', 'unknown'))
            if candidate.content and candidate.content.parts:
                full_text = "".join(
                    part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text
                )
        
        if not full_text:
            full_text = response.text or ""
        
        logger.info("gemini_response", finish_reason=finish_reason, text_length=len(full_text))

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return CompletionResult(
            content=full_text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def estimate_token_count(self, text: str, *, model: str = "gemini-3-flash-preview") -> int:
        """Provide a simple character-based heuristic for token estimation."""
        return len(text) // 4
