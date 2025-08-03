import asyncio
import os
from typing import Dict, List

from agentic_ai.llm import (
    LLMConfig,
    LLMMessage,
    LLMProvider,
    OpenRouterProvider,
    UnifiedLLMClient,
)


class PrecisionAgent:
    """Agent that needs precise control over message formatting and metadata."""

    def __init__(self, unified_client: UnifiedLLMClient, config: LLMConfig):
        self.client = unified_client
        self.config = config
        self.message_history: List[LLMMessage] = []

    async def structured_conversation(
        self, user_input: str, system_prompt: str
    ) -> Dict:
        """Demonstrate precise message control with metadata extraction."""

        # Build messages with precise control
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_input, name="user_001"),
        ]

        # Get response with full metadata
        response = await self.client.complete(messages, self.config)

        # Store in history for potential future use
        self.message_history.extend(messages)
        self.message_history.append(
            LLMMessage(role="assistant", content=response.content)
        )

        return {
            "response": response.content,
            "model": response.model,
            "provider": response.provider.value,
            "usage": response.usage,
            "metadata": response.metadata,
        }

    async def multi_turn_with_metadata(self, turns: List[Dict[str, str]]) -> List[Dict]:
        """Handle multi-turn conversation with detailed tracking."""

        results = []
        messages = []

        for turn in turns:
            # Add user message
            user_msg = LLMMessage(
                role="user", content=turn["content"], name=turn.get("name", "user")
            )
            messages.append(user_msg)

            # Get response
            response = await self.client.complete(messages, self.config)

            # Add assistant response to messages
            assistant_msg = LLMMessage(role="assistant", content=response.content)
            messages.append(assistant_msg)

            # Track results
            results.append(
                {
                    "turn": len(results) + 1,
                    "user_input": turn["content"],
                    "response": response.content,
                    "usage": response.usage,
                    "metadata": response.metadata,
                }
            )

        return results


class MultiProviderComparisonAgent:
    """Agent that compares responses across multiple providers."""

    def __init__(self, unified_client: UnifiedLLMClient):
        self.client = unified_client
        self.provider_configs = {}

    def add_provider(self, provider_name: str, config: LLMConfig):
        """Add a provider configuration."""
        self.provider_configs[provider_name] = config

    async def compare_providers(self, prompt: str, system_context: str = None) -> Dict:
        """Send same prompt to multiple providers and compare results."""

        # Build messages
        messages = []
        if system_context:
            messages.append(LLMMessage(role="system", content=system_context))
        messages.append(LLMMessage(role="user", content=prompt))

        results = {}

        for provider_name, config in self.provider_configs.items():
            try:
                response = await self.client.complete(messages, config)
                results[provider_name] = {
                    "content": response.content,
                    "model": response.model,
                    "provider": response.provider.value,
                    "usage": response.usage,
                    "success": True,
                }
            except Exception as e:
                results[provider_name] = {"error": str(e), "success": False}

        return results


class TokenOptimizedAgent:
    """Agent that carefully manages token usage for cost optimization."""

    def __init__(self, unified_client: UnifiedLLMClient, config: LLMConfig):
        self.client = unified_client
        self.config = config
        self.total_tokens_used = 0

    async def token_aware_completion(self, messages: List[LLMMessage]) -> Dict:
        """Complete with detailed token tracking."""

        # Estimate input tokens (rough approximation)
        estimated_input_tokens = sum(len(msg.content.split()) * 1.3 for msg in messages)

        response = await self.client.complete(messages, self.config)

        # Track usage
        if response.usage:
            self.total_tokens_used += response.usage.get("total_tokens", 0)

        return {
            "response": response.content,
            "estimated_input_tokens": int(estimated_input_tokens),
            "actual_usage": response.usage,
            "cumulative_tokens": self.total_tokens_used,
            "model": response.model,
        }

    async def summarize_conversation(self, conversation: List[LLMMessage]) -> str:
        """Summarize a long conversation to save tokens."""

        # Create summarization messages
        conversation_text = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in conversation]
        )

        summary_messages = [
            LLMMessage(
                role="system",
                content="Summarize the following conversation concisely, preserving key information.",  # noqa: E501
            ),
            LLMMessage(role="user", content=conversation_text),
        ]

        result = await self.token_aware_completion(summary_messages)
        return result["response"]


class StreamingAgent:
    """Agent that handles streaming responses (when supported)."""

    def __init__(self, unified_client: UnifiedLLMClient, config: LLMConfig):
        self.client = unified_client
        self.config = config

    async def non_streaming_with_progress(self, messages: List[LLMMessage]) -> Dict:
        """Simulate progress tracking for non-streaming responses."""

        print("🤔 Thinking...")
        response = await self.client.complete(messages, self.config)
        print("✅ Response ready!")

        return {
            "response": response.content,
            "model": response.model,
            "usage": response.usage,
        }


async def demonstrate_low_level_patterns():
    """Demonstrate low-level usage patterns with UnifiedLLMClient."""

    # Check for API key
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("Error: OPEN_ROUTER_API_KEY environment variable not set.")
        return

    # Set up unified client
    client = UnifiedLLMClient()

    # Configure OpenRouter provider
    openrouter_config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=512,
        api_key=api_key,
    )

    provider = OpenRouterProvider(openrouter_config)
    client.register_provider(provider)

    print("=== Precision Agent (Structured Message Control) ===")
    precision_agent = PrecisionAgent(client, openrouter_config)

    result = await precision_agent.structured_conversation(
        user_input="Explain quantum computing in simple terms",
        system_prompt="You are a physics teacher explaining complex topics to high school students.",  # noqa: E501
    )

    print(f"Response: {result['response'][:100]}...")
    print(f"Model: {result['model']}")
    print(f"Usage: {result['usage']}")
    print()

    print("=== Multi-turn with Metadata Tracking ===")
    turns = [
        {"content": "What is machine learning?", "name": "student_1"},
        {
            "content": "How does it differ from traditional programming?",
            "name": "student_1",
        },
        {"content": "Give me a simple example", "name": "student_1"},
    ]

    multi_turn_results = await precision_agent.multi_turn_with_metadata(turns)

    for result in multi_turn_results:
        print(f"Turn {result['turn']}: {result['user_input']}")
        print(f"Response: {result['response'][:80]}...")
        print(f"Usage: {result['usage']}")
        print()

    print("=== Token Optimization Agent ===")
    token_agent = TokenOptimizedAgent(client, openrouter_config)

    # Example conversation to summarize
    long_conversation = [
        LLMMessage(role="user", content="Tell me about renewable energy"),
        LLMMessage(
            role="assistant",
            content="Renewable energy comes from natural sources that replenish themselves...",  # noqa: E501
        ),
        LLMMessage(role="user", content="What about solar power specifically?"),
        LLMMessage(
            role="assistant",
            content="Solar power harnesses energy from the sun using photovoltaic panels...",  # noqa: E501
        ),
        LLMMessage(role="user", content="How efficient are modern solar panels?"),
        LLMMessage(
            role="assistant",
            content="Modern solar panels typically achieve 15-22% efficiency...",
        ),
    ]

    summary = await token_agent.summarize_conversation(long_conversation)
    print(f"Conversation summary: {summary}")
    print(f"Total tokens used by agent: {token_agent.total_tokens_used}")
    print()

    print("=== Direct Message Control ===")
    # Direct usage without agent wrapper
    messages = [
        LLMMessage(
            role="system",
            content="You are a helpful assistant that responds in JSON format.",
        ),
        LLMMessage(
            role="user", content="List 3 benefits of exercise", name="health_seeker"
        ),
    ]

    direct_response = await client.complete(messages, openrouter_config)
    print(f"Direct response: {direct_response.content}")
    print(f"Provider: {direct_response.provider}")
    print(f"Model: {direct_response.model}")
    print()

    print("=== Health Check ===")
    health_status = await client.health_check_all()
    print(f"Provider health: {health_status}")


if __name__ == "__main__":
    asyncio.run(demonstrate_low_level_patterns())
