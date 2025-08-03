import os
from typing import List

from ...open_router import AIAgent
from ...utils.core.schemas import ExtraResponseSettings
from ..interface import BaseLLMProvider, LLMConfig, LLMMessage, LLMProvider, LLMResponse


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider implementation."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Get API key from config or environment (use the correct env var name)
        api_key = config.api_key or os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key required")

        # Create default ExtraResponseSettings with reasonable defaults
        default_settings = ExtraResponseSettings(
            temperature=config.temperature,
            max_tokens=config.max_tokens or 4096,
            tool_choice="auto"
        )

        # Initialize the existing OpenRouter client with correct parameters
        self._client = AIAgent(
            agent_name=f"OpenRouter-{config.model}",
            model_name=config.model,
            sys_instructions=None,
            tools=[],
            extra_response_settings=default_settings
        )

    def _messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Convert list of messages to a single prompt string."""
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
            else:
                prompt_parts.append(f"{msg.role}: {msg.content}")

        return "\n\n".join(prompt_parts)

    async def complete(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenRouter."""
        try:
            # Convert messages to single prompt string
            prompt_string = self._messages_to_prompt(messages)

            # Call the AIAgent's prompt method with single string
            agent_response = await self._client.prompt(message=prompt_string)

            # Convert to unified format
            return LLMResponse(
                content=agent_response.final_response,
                model=self.config.model,
                provider=LLMProvider.OPENROUTER,
                usage=getattr(agent_response, 'usage', None),
                metadata={
                    "parsed_response": agent_response.parsed_response,
                    "raw_response": agent_response
                }
            )

        except Exception as e:
            raise RuntimeError(f"OpenRouter completion failed: {str(e)}") from e

    async def health_check(self) -> bool:
        """Check OpenRouter API health."""
        try:
            # Simple health check with minimal request
            test_response = await self._client.prompt(message="Hi")
            return bool(test_response.final_response)
        except Exception:
            return False