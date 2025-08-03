from typing import Any, Dict, List, Optional

from .interface import LLMConfig, LLMMessage, UnifiedLLMClient


class AgentLLM:
    """Simplified LLM interface for agents."""

    def __init__(self, unified_client: UnifiedLLMClient, default_config: LLMConfig):
        self.client = unified_client
        self.default_config = default_config

    async def think(self, prompt: str, context: Optional[str] = None) -> str:
        """Simple text-in, text-out interface for agents."""
        messages = []
        if context:
            messages.append(LLMMessage(role="system", content=context))
        messages.append(LLMMessage(role="user", content=prompt))

        response = await self.client.complete(messages, self.default_config)
        return response.content

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat interface with message history."""
        llm_messages = [
            LLMMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        response = await self.client.complete(llm_messages, self.default_config)
        return response.content