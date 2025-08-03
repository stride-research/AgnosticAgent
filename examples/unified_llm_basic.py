import asyncio
import os

from agentic_ai.llm import (
    AgentLLM,
    LLMConfig,
    LLMProvider,
    OpenRouterProvider,
    UnifiedLLMClient,
)


async def main():
    """Demonstrate basic unified LLM usage."""

    # Check for API key (using the correct variable name the provider expects)
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("Error: OPEN_ROUTER_API_KEY environment variable not set.")
        print("Please set it by running:")
        print("export OPEN_ROUTER_API_KEY='your-api-key-here'")
        print("Get your API key from: https://openrouter.ai/keys")
        return

    # Create unified client
    client = UnifiedLLMClient()

    # Configure OpenRouter provider with explicit max_tokens
    openrouter_config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1024,  # Set a reasonable limit
        api_key=api_key,
    )

    # Register provider
    openrouter_provider = OpenRouterProvider(openrouter_config)
    client.register_provider(openrouter_provider)

    # Create agent interface
    agent_llm = AgentLLM(client, openrouter_config)

    # Simple usage
    response = await agent_llm.think("What is the capital of France?")
    print(f"Response: {response}")

    # Usage with context
    response = await agent_llm.think(
        "What is 2+2?", context="You are a helpful math tutor. Be encouraging."
    )
    print(f"Math response: {response}")

    # Health check
    health_status = await client.health_check_all()
    print(f"Provider health: {health_status}")


if __name__ == "__main__":
    asyncio.run(main())
