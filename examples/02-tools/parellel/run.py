from __future__ import annotations

import asyncio
import logging

from radical.asyncflow import WorkflowEngine, ThreadExecutionBackend

from agentic_ai import AcademyWorkflowIntegration
from agentic_ai.utils import add_context_to_log
from .utils.agents import WeatherGuy

logger = logging.getLogger(__name__)

async def main() -> int:

      # Create workflow engine
      backend = ThreadExecutionBackend({})
      backend.set_main_loop()
      flow = WorkflowEngine(backend=backend)

      async with AcademyWorkflowIntegration(flow) as manager:
            with add_context_to_log(agent_id=WeatherGuy.__name__):
                  get_weather_task = manager.create_agent_task(
                        WeatherGuy, "get_weather"
                  )

                  @flow.function_task
                  async def get_weather_block():
                        logger.debug("Starting weather block")
                        return await get_weather_task()
                  
                  response = await get_weather_block()
                  logger.debug(f"Final get_weather_task: {response}")

      await flow.shutdown()
      return 0

if __name__ == "__main__":
      asyncio.run(main())