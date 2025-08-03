import asyncio
import os
from typing import Dict, List

from agentic_ai.llm import (
    AgentLLM,
    LLMConfig,
    LLMProvider,
    OpenRouterProvider,
    UnifiedLLMClient,
)


class SimpleReasoningAgent:
    """A simple agent that uses the unified LLM interface for reasoning tasks."""

    def __init__(self, agent_llm: AgentLLM, name: str = "ReasoningAgent"):
        self.llm = agent_llm
        self.name = name
        self.conversation_history: List[Dict[str, str]] = []

    async def reason_about(self, problem: str) -> str:
        """Use the LLM to reason about a problem step by step."""
        context = "You are a logical reasoning assistant. Think step by step and show your work."  # noqa: E501

        response = await self.llm.think(problem, context=context)

        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": problem})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    async def continue_conversation(self, follow_up: str) -> str:
        """Continue the conversation with context from previous exchanges."""
        # Add the new message to history
        self.conversation_history.append({"role": "user", "content": follow_up})

        # Use the chat method for multi-turn conversations
        response = await self.llm.chat(self.conversation_history)

        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response


class MultiProviderAgent:
    """An agent that can switch between different LLM providers based on task
    type."""

    def __init__(self, unified_client: UnifiedLLMClient):
        self.client = unified_client
        self.provider_configs = {}

    def add_provider_config(self, provider: LLMProvider, config: LLMConfig):
        """Add a provider configuration for specific tasks."""
        self.provider_configs[provider] = config

    async def generate_creative_content(self, prompt: str) -> str:
        """Use a creative model for content generation."""
        # Use OpenRouter with higher temperature for creativity
        config = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model="gpt-4",
            temperature=0.9,
            max_tokens=2048,
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        )

        agent_llm = AgentLLM(self.client, config)
        context = "You are a creative writer. Be imaginative and original."

        return await agent_llm.think(prompt, context=context)

    async def analyze_data(self, data_description: str) -> str:
        """Use a logical model for data analysis."""
        # Use OpenRouter with lower temperature for analysis
        config = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1024,
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        )

        agent_llm = AgentLLM(self.client, config)
        context = "You are a data analyst. Be precise and factual in your analysis."  # noqa: E501

        return await agent_llm.think(data_description, context=context)


class TaskOrchestrationAgent:
    """An agent that breaks down complex tasks and uses LLMs for each subtask."""

    def __init__(self, agent_llm: AgentLLM):
        self.llm = agent_llm
        self.task_results: Dict[str, str] = {}

    async def plan_task(self, complex_task: str) -> List[str]:
        """Break down a complex task into subtasks."""
        planning_prompt = f"""
        Break down this complex task into 3-5 smaller, manageable subtasks:

        Task: {complex_task}

        Return each subtask on a new line, numbered 1-5.
        """

        context = "You are a task planning expert. Break down complex problems logically."  # noqa: E501
        response = await self.llm.think(planning_prompt, context=context)

        # Parse the response to extract subtasks
        subtasks = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering and clean up
                subtask = line.split(".", 1)[-1].strip()
                if subtask:
                    subtasks.append(subtask)

        return subtasks

    async def execute_subtask(self, subtask: str, task_context: str = "") -> str:
        """Execute a specific subtask."""
        full_context = f"You are executing a subtask. {task_context}".strip()

        result = await self.llm.think(subtask, context=full_context)
        self.task_results[subtask] = result
        return result

    async def synthesize_results(self, original_task: str) -> str:
        """Combine results from all subtasks into a final answer."""
        results_summary = "\n".join(
            [
                f"Subtask: {task}\nResult: {result}\n"
                for task, result in self.task_results.items()
            ]
        )

        synthesis_prompt = f"""
        Original task: {original_task}

        Subtask results:
        {results_summary}

        Synthesize these results into a comprehensive final answer.
        """

        context = "You are synthesizing multiple pieces of information into a coherent final answer."  # noqa: E501
        return await self.llm.think(synthesis_prompt, context=context)


async def demonstrate_agent_patterns():
    """Demonstrate different ways agents can use the unified LLM interface."""

    # Check for API key
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        print("Error: OPEN_ROUTER_API_KEY environment variable not set.")
        return

    # Set up unified client
    client = UnifiedLLMClient()

    # Configure and register OpenRouter provider
    config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=1024,
        api_key=api_key,
    )

    provider = OpenRouterProvider(config)
    client.register_provider(provider)

    # Create agent interface
    agent_llm = AgentLLM(client, config)

    print("=== Simple Reasoning Agent ===")
    reasoning_agent = SimpleReasoningAgent(agent_llm)

    # Single reasoning task
    problem = "If I have 5 apples and give away 2, then buy 3 more, how many do I have?"  # noqa: E501
    result = await reasoning_agent.reason_about(problem)
    print(f"Problem: {problem}")
    print(f"Result: {result}\n")

    # Follow-up question
    follow_up = "What if I then eat 1 apple?"
    follow_result = await reasoning_agent.continue_conversation(follow_up)
    print(f"Follow-up: {follow_up}")
    print(f"Result: {follow_result}\n")

    print("=== Multi-Provider Agent ===")
    multi_agent = MultiProviderAgent(client)

    # Creative task
    creative_result = await multi_agent.generate_creative_content(
        "Write a short poem about artificial intelligence"
    )
    print(f"Creative content: {creative_result}\n")

    # Analytical task
    analysis_result = await multi_agent.analyze_data(
        "Analyze the trend: Website traffic increased 25% in Q1, 15% in Q2, 5% in Q3"
    )
    print(f"Data analysis: {analysis_result}\n")

    print("=== Task Orchestration Agent ===")
    orchestrator = TaskOrchestrationAgent(agent_llm)

    # Complex task
    complex_task = (
        "Plan a sustainable office space for a 50-person tech startup"  # noqa: E501
    )

    # Break down the task
    subtasks = await orchestrator.plan_task(complex_task)
    print(f"Subtasks for '{complex_task}':")
    for i, subtask in enumerate(subtasks, 1):
        print(f"  {i}. {subtask}")
    print()

    # Execute each subtask
    print("Executing subtasks...")
    for subtask in subtasks[:2]:  # Execute first 2 for demo
        result = await orchestrator.execute_subtask(
            subtask, "Focus on sustainability and tech startup needs."
        )
        print(f"Subtask: {subtask}")
        print(f"Result: {result[:100]}...\n")

    # Synthesize results
    if orchestrator.task_results:
        final_result = await orchestrator.synthesize_results(complex_task)
        print(f"Final synthesis: {final_result[:200]}...\n")


if __name__ == "__main__":
    asyncio.run(demonstrate_agent_patterns())
    asyncio.run(demonstrate_agent_patterns())
