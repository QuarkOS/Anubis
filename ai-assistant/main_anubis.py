"""Entry point for the Anubis multimodal desktop assistant.

Wires together the domain layer (AgentService, LLM providers, Memory)
and the I/O layer (AudioVisionService for hotkey/recording/TTS).
"""

import asyncio
import os
import structlog

from anubis.config import get_settings
from anubis.prompts import PromptRegistry
from anubis.domain.schemas import ChatRequest
from anubis.repositories.conversation_memory import InMemoryConversationRepository
from anubis.repositories.llm_gemini import LLMGeminiProvider
from anubis.services.agent import AgentService
from anubis.services.context import SlidingWindowContextBuilder
from anubis.services.io_service import AudioVisionService

logger = structlog.get_logger()

settings = get_settings()
if not settings.llm_api_key:
    logger.error("Missing ANUBIS_LLM_API_KEY in environment. Terminating.")
    exit(1)

llm_provider = LLMGeminiProvider(api_key=settings.llm_api_key)
repo = InMemoryConversationRepository()
context_builder = SlidingWindowContextBuilder(llm=llm_provider)
prompts = PromptRegistry()

prompts._templates["system.default"] = (
    "You are Anubis, a highly intelligent voice-first desktop assistant. "
    "You communicate via speech. By default, your answers MUST BE extremely concise: never exceed 1-2 short sentences. "
    "ONLY provide a detailed explanation IF the user explicitly asks you to 'explain' or 'elaborate'. "
    "NEVER use markdown, bullet points, or complex formatting. Just give the direct answer. "
    "The user is sending you audio recordings of their voice and a screenshot of their current screen. "
    "Use the screenshot to answer questions about what they are doing or looking at. "
    "IMPORTANT LANGUAGE RULE: Always respond in the detected language of the user. "
    "However, if the user is speaking a regional dialect (e.g., Austrian German), you must reply in the standard base language (e.g., standard High German)."
)

agent_service = AgentService(
    llm=llm_provider,
    repo=repo,
    context_builder=context_builder,
    prompts=prompts
)
io_service = AudioVisionService(hotkey='f14')

current_conversation_id = None


async def process_captured_media(audio_path: str, vision_path: str) -> None:
    """Handle audio and vision input captured by the I/O layer and generate a response."""
    global current_conversation_id
    logger.info("input_ready", audio=audio_path, vision=vision_path)
    
    prompt_text = f"<file:{audio_path}> <file:{vision_path}>"
    
    logger.info("anubis_thinking")
    try:
        req = ChatRequest(
            conversation_id=current_conversation_id,
            message=prompt_text,
            model="gemini-3-flash-preview",
            temperature=0.7,
            max_tokens=1000
        )
        response = await agent_service.chat(req)
        current_conversation_id = response.conversation_id
        
        logger.info("anubis_reply", text=response.reply)
        await io_service.speak(response.reply)
        
    except Exception as e:
        logger.error("anubis_error", error=str(e))
        await io_service.speak("I'm sorry, I encountered an error processing that.")


def handle_input_ready_sync(audio_path: str, vision_path: str) -> None:
    """Bridge synchronous I/O callbacks to the async processing pipeline."""
    asyncio.run(process_captured_media(audio_path, vision_path))


async def main() -> None:
    """Initialize the assistant and start background input listeners."""
    logger.info("=======================================")
    logger.info("🌟 ANUBIS VOICE ASSISTANT ONLINE")
    logger.info("=======================================")
    logger.info(f"Press and hold '{io_service.hotkey}' to speak.")
    
    io_service.listen_in_background(handle_input_ready_sync)
    
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Anubis...")
