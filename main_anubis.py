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
from anubis.repositories.system_windows import WindowsSystemProbe
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
system_probe = WindowsSystemProbe()
prompts = PromptRegistry()

agent_service = AgentService(
    llm=llm_provider,
    repo=repo,
    context_builder=context_builder,
    prompts=prompts,
    system_probe=system_probe
)
io_service = AudioVisionService(hotkey='f14')

current_conversation_id = None


async def process_captured_media(audio_path: str, vision_path: str) -> None:
    """
    Handle multimodal input from the I/O layer and generate an assistant response.
    
    This function acts as the primary pipeline, uploading media to Gemini, 
    managing conversation state, and triggering the local TTS engine.
    """
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
            max_tokens=4096
        )
        response = await agent_service.chat(req)
        current_conversation_id = response.conversation_id
        
        logger.info("raw_gemini_response", text=response.reply)
        logger.info("anubis_reply", text=response.reply)
        await io_service.speak(response.reply)
        
    except Exception as e:
        logger.error("anubis_error", error=str(e))
        await io_service.speak("I'm sorry, I encountered an error processing that.")


def handle_input_ready_sync(audio_path: str, vision_path: str) -> None:
    """Synchronous bridge callback required by the background listening thread."""
    asyncio.run(process_captured_media(audio_path, vision_path))


async def main() -> None:
    """Initialize the service container and start the background multimodal listeners."""
    logger.info("=======================================")
    logger.info("🌟 ANUBIS MULTIMODAL ASSISTANT ONLINE")
    logger.info("=======================================")
    logger.info("Wake Word: 'Anubis'")
    logger.info("Status: Listening in background...")

    # Start the listening loop
    io_service.listen_in_background(handle_input_ready_sync)

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Anubis...")
