"""Entry point for the Anubis multimodal desktop assistant.

Wires together the domain layer (AgentService, LLM providers, Memory)
and the I/O layer (AudioVisionService for hotkey/recording/TTS).
"""

import asyncio
import os
import structlog

# Fix for potential OpenMP/Torch runtime conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from anubis.config import get_settings
from anubis.prompts import PromptRegistry
from anubis.domain.schemas import ChatRequest
from anubis.repositories.conversation_memory import InMemoryConversationRepository
from anubis.repositories.llm_gemini import LLMGeminiProvider
from anubis.repositories.system_windows import WindowsSystemProbe
from anubis.services.agent import AgentService
from anubis.services.context import SlidingWindowContextBuilder
from anubis.services.io_service import AudioVisionService
from anubis.services.visual_feedback import VisualFeedbackService

logger = structlog.get_logger()

# Global pointers to services initialized in main()
current_conversation_id = None
agent_service = None
io_service = None
visual_feedback = None


async def process_captured_media(audio_path: str, vision_path: str) -> None:
    """
    Handle multimodal input from the I/O layer and generate an assistant response.
    
    This function acts as the primary pipeline, uploading media to Gemini, 
    managing conversation state, and triggering the local TTS engine.
    """
    global current_conversation_id
    logger.info("input_ready", audio=audio_path, vision=vision_path)
    
    # Transition to 'thinking' (Cyan Aura)
    if visual_feedback:
        visual_feedback.set_state("thinking")
    
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
        
        # Return to idle before speaking
        if visual_feedback:
            visual_feedback.set_state("idle")
        await io_service.speak(response.reply)
        
    except Exception as e:
        if visual_feedback:
            visual_feedback.set_state("idle")
        logger.error("anubis_error", error=str(e))
        await io_service.speak("I'm sorry, I encountered an error processing that.")


def handle_wake_word_feedback() -> None:
    """Provide immediate visual acknowledgement when the wake word is confirmed."""
    if visual_feedback:
        visual_feedback.set_state("waking")


def handle_input_ready_sync(audio_path: str, vision_path: str) -> None:
    """Synchronous bridge callback required by the background listening thread."""
    asyncio.run(process_captured_media(audio_path, vision_path))


async def main() -> None:
    """Initialize the service container and start the background multimodal listeners."""
    global agent_service, io_service, visual_feedback
    
    logger.info("=======================================")
    logger.info("* ANUBIS MULTIMODAL ASSISTANT ONLINE *")
    logger.info("=======================================")
    
    settings = get_settings()
    if not settings.llm_api_key:
        logger.error("Missing ANUBIS_LLM_API_KEY in environment. Terminating.")
        return

    # Initialize services inside main to ensure thread/context safety
    logger.info("initializing_services")
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

    wake_label = getattr(io_service, 'wake_word_label', 'Unknown')
    logger.info(f"Wake Word: '{wake_label}' active")
    logger.info("Status: Listening in background...")

    # Start the listening loop with immediate wake-word feedback
    io_service.listen_in_background(
        callback=handle_input_ready_sync, 
        on_wake_word=handle_wake_word_feedback
    )

    while True:
        await asyncio.sleep(1)


def run_background_loop():
    asyncio.run(main())


if __name__ == "__main__":
    import sys
    import threading
    from PyQt6.QtWidgets import QApplication
    
    # 1. Create Qt application in main thread
    app = QApplication(sys.argv)
    
    # 2. Instantiate UI components in main thread
    visual_feedback = VisualFeedbackService()
    
    try:
        # 3. Start Anubis domain logic in a background thread
        t = threading.Thread(target=run_background_loop, daemon=True)
        t.start()
        
        # 4. Run Qt Event Loop on the main thread
        sys.exit(app.exec())
    except KeyboardInterrupt:
        logger.info("Shutting down Anubis...")
