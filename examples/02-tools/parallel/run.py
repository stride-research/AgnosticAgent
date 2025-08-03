from __future__ import annotations

import asyncio
import argparse
import logging

from .utils.toolkit import WeatherToolkit
from agentic_ai import LLMAgent


"""Temperature example """


logger = logging.getLogger(__name__)


async def run_example(backend="ollama", model="qwen3:8b"):
      agent = LLMAgent(
                  llm_backend=backend,
                  agent_name="WeatherGuy",
                  model_name=model,
                  tools=WeatherToolkit().extract_tools_names()
            )

      response = await agent.prompt(f"What is the weather like in San Francisco (temperature and humidity)?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openrouter", "openai"])
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args()
    
    asyncio.run(run_example(backend=args.backend, model=args.model))
