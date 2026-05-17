import pytest
from anubis.services.agent import AgentService
from anubis.domain.schemas import SystemState, ChatRequest, ProcessInfo, GPUInfo, NetworkState, UserContext
from anubis.domain.protocols import SystemProbe, LLMProvider, ConversationRepository, ContextBuilder
from anubis.domain.models import CompletionResult, Message, Conversation
from anubis.prompts import PromptRegistry
from unittest.mock import MagicMock

class MockSystemProbe(SystemProbe):
    """Stub implementation of SystemProbe for unit testing AgentService logic."""

    async def probe_state(self) -> SystemState:
        return SystemState(
            cpu_percent=10.0,
            memory_percent=50.0,
            battery_percent=100.0,
            active_window="Test Window",
            top_processes=[
                ProcessInfo(name="Test Process", cpu_percent=5.0, memory_mb=100.0, pid=123)
            ],
            gpus=[
                GPUInfo(name="Test GPU", load_percent=20.0, memory_used_mb=1000.0, memory_total_mb=8000.0, temperature=50.0)
            ],
            network=NetworkState(ssid="TestWiFi", upload_kbps=100.0, download_kbps=500.0, public_ip="1.1.1.1", location="Test City"),
            user_context=UserContext(clipboard_preview="Test Clipboard", recent_files=["file1", "file2"], media_info="Spotify: Song"),
            os_name="Test OS",
            timestamp="2026-05-17T12:00:00Z"
        )

class FakeLLM:
    """Minimal LLM provider stub to avoid real API calls during testing."""

    async def generate_response(self, *args, **kwargs):
        return CompletionResult(content="Reply", model="test", prompt_tokens=1, completion_tokens=1)
    async def estimate_token_count(self, text, **kwargs):
        return len(text)

@pytest.mark.asyncio
async def test_agent_chat_with_system_probe():
    # Setup
    llm = FakeLLM()
    repo = MagicMock(spec=ConversationRepository)
    repo.fetch_conversation.return_value = None
    ctx = MagicMock(spec=ContextBuilder)
    ctx.build_message_context.return_value = []
    
    prompts = MagicMock(spec=PromptRegistry)
    prompts.render_prompt.return_value = "System Context: Test Window"
    
    probe = MockSystemProbe()
    service = AgentService(
        llm=llm,
        repo=repo,
        context_builder=ctx,
        prompts=prompts,
        system_probe=probe
    )
    
    # Execute
    request = ChatRequest(message="Hello")
    await service.chat(request)
    
    # Verify
    assert prompts.render_prompt.called
    found_system_state = False
    for call in prompts.render_prompt.call_args_list:
        if "system_state" in call.kwargs:
            state = call.kwargs["system_state"]
            assert state.active_window == "Test Window"
            assert len(state.top_processes) == 1
            assert state.gpus[0].name == "Test GPU"
            assert state.network.ssid == "TestWiFi"
            assert state.user_context.clipboard_preview == "Test Clipboard"
            found_system_state = True
    assert found_system_state
