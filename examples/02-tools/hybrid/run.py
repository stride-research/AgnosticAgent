from __future__ import annotations

import asyncio
import logging

from .utils.toolkit import ChefToolkit
from agentic_ai import AIAgent


logger = logging.getLogger(__name__)


async def run_example():
      LLMAgent = AIAgent(
                  agent_name="ChefAssistant",
                  sys_instructions="Given a dish you need to first provide the ingredients required, then return the price of thesee ingredients. Use the provided tools.\
                  Consider getting the price per ingredients in parallel  ",
                  tools=ChefToolkit().extract_tools_names()
            )

      response = await LLMAgent.prompt(f"What do you need for pizza and what is the price of the ingredients?")
      logger.debug(f"Response is: {response}")

if __name__ == "__main__":
    asyncio.run(run_example())
