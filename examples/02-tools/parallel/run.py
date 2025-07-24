from __future__ import annotations

import asyncio
import logging

from .utils.toolkit import WeatherToolkit
from agentic_ai import AIAgent


"""Temperature example """


logger = logging.getLogger(__name__)


def run_example():
      LLMAgent = AIAgent(
                  agent_name="WeatherGuy",
                  model_name="google/gemini-2.0-flash-001",
                  tools=WeatherToolkit().extract_tools_names()
            )

      response = LLMAgent.prompt(f"What is the weather like in San Francisco (temperature and humidity)?")

if __name__ == "__main__":
    run_example()

