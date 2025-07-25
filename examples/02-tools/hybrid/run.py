from __future__ import annotations

import asyncio
import logging

from .utils import toolkit
from agentic_ai import AIAgent


"""Temperature example """


logger = logging.getLogger(__name__)


async def run_example():
      LLMAgent = AIAgent(
                  agent_name="ChefAssistant",
                  sys_instructions="Given a dish you need to first provide the ingredients required, then return the price of thesee ingredients. Use the provided tools.\
                  For get_ingredient_price tool you may need to call it several times in parallel with different ingredients.  "
            )

      response = await LLMAgent.prompt(f"What do you need for pizza?")
      logger.debug(f"Response is: {response}")

if __name__ == "__main__":
    asyncio.run(run_example())
