"""Sliding-window context builder.

Implements the ContextBuilder protocol.  Trims conversation history from
the oldest messages forward until the total token budget is satisfied,
guaranteeing the system prompt and most recent user message are always
included.
"""

from __future__ import annotations

from anubis.domain.models import Conversation, Message, Role
from anubis.domain.protocols import LLMProvider


class SlidingWindowContextBuilder:
    """Trims conversation history to fit within a specified token budget, prioritizing recent messages."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def build_message_context(
        self,
        conversation: Conversation,
        *,
        system_prompt: str,
        max_context_tokens: int,
    ) -> list[Message]:
        """Assemble a list of messages that fits within the token budget by sliding the history window."""
        system_msg = Message(role=Role.SYSTEM, content=system_prompt)
        system_tokens = await self._llm.estimate_token_count(system_prompt)

        history = [m for m in conversation.messages if m.role != Role.SYSTEM]
        if not history:
            return [system_msg]

        latest = history[-1]
        latest_tokens = await self._llm.estimate_token_count(latest.content)

        remaining_budget = max_context_tokens - (system_tokens + latest_tokens)

        if remaining_budget < 0:
            return [system_msg, latest]

        selected: list[Message] = []
        for msg in reversed(history[:-1]):
            msg_tokens = await self._llm.estimate_token_count(msg.content)
            if msg_tokens > remaining_budget:
                break
            selected.append(msg)
            remaining_budget -= msg_tokens

        selected.reverse()
        return [system_msg, *selected, latest]
