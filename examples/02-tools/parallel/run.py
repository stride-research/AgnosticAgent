from __future__ import annotations

import asyncio
import logging

from .utils.toolkit import WeatherToolkit
from agentic_ai import LLMAgent


"""Temperature example """


logger = logging.getLogger(__name__)


async def run_example():
      agent = LLMAgent(
                  llm_backend="ollama",
                  agent_name="WeatherGuy",
                  model_name="qwen3:8b",
                  tools=WeatherToolkit().extract_tools_names()
            )

      response = await agent.prompt(f"What is the weather like in San Francisco (temperature and humidity)?")

if __name__ == "__main__":
    asyncio.run(run_example())
