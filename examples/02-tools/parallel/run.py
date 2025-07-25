from __future__ import annotations

import asyncio
import logging

from .utils import toolkit
from agentic_ai import AIAgent


"""Temperature example """


logger = logging.getLogger(__name__)


async def run_example():
      LLMAgent = AIAgent(
                  agent_name="WeatherGuy",
                  model_name="google/gemini-2.0-flash-001",
            )

      response = await LLMAgent.prompt(f"What is the weather like in San Francisco (temperature and humidity)?")

if __name__ == "__main__":
    asyncio.run(run_example())
