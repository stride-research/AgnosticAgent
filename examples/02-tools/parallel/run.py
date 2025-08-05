from __future__ import annotations

import asyncio
import argparse
import logging

from .utils.toolkit import WeatherToolkit
from agentic_ai import LLMAgent
from ...config import inline_args



"""Temperature example """


logger = logging.getLogger(__name__)


async def run_example(backend:str, model:str):
      agent = LLMAgent(
                  llm_backend=backend,
                  agent_name="WeatherGuy",
                  model_name=model,
                  tools=WeatherToolkit().extract_tools_names()
            )

      response = await agent.prompt(f"What is the weather like in San Francisco (temperature and humidity)?")

if __name__ == "__main__":
    asyncio.run(run_example(backend=inline_args.backend, 
                            model=inline_args.model))
