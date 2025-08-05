from __future__ import annotations

import asyncio
import argparse
import logging

from .utils.toolkit import ChefToolkit
from agentic_ai import LLMAgent

from ...config import inline_args



logger = logging.getLogger(__name__)


async def run_example(backend:str, model:str):
      agent = LLMAgent(
                  llm_backend=backend,
                  agent_name="ChefAssistant",
                  model_name=model,
                  sys_instructions="Given a dish you need to first provide the ingredients required, then return the price of thesee ingredients. Use the provided tools.\
                  For get_ingredient_price tool you may need to call it several times in parallel with different ingredients.  ",
                  tools=ChefToolkit().extract_tools_names()
            )

      response = await agent.prompt(f"What do you need for pizza?")

if __name__ == "__main__":
    asyncio.run(run_example(backend=inline_args.backend,
                             model=inline_args.model))
