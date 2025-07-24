from __future__ import annotations

import asyncio
import logging

from radical.asyncflow import WorkflowEngine, ThreadExecutionBackend

from agentic_ai import AcademyWorkflowIntegration
from agentic_ai.utils import add_context_to_log
from .utils.agents import Mathematician

logger = logging.getLogger(__name__)

async def main() -> int:

      # Create workflow engine
      backend = ThreadExecutionBackend({})
      backend.set_main_loop()
      flow = WorkflowEngine(backend=backend)

      async with AcademyWorkflowIntegration(flow) as manager:
            with add_context_to_log(agent_id=Mathematician.__name__):
                  do_math_task_ = manager.create_agent_task(
                        Mathematician, "basic_math"
                  )

                  @flow.function_task
                  async def do_math_block():
                        logger.debug("Starting math block")
                        return await do_math_task_()
                  
                  response = await do_math_block()
                  logger.debug(f"Final response: {response}")

      await flow.shutdown()
      return 0

if __name__ == "__main__":
      asyncio.run(main())