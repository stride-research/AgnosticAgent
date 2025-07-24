import logging
import asyncio
from typing import List

from utils.agents import FileIngestor

from radical.asyncflow import WorkflowEngine, ThreadExecutionBackend
from agentic_ai import AcademyWorkflowIntegration
from agentic_ai.utils import add_context_to_log

logger = logging.getLogger(__name__)

async def main() -> int:

      backend = ThreadExecutionBackend({})
      backend.set_main_loop()
      flow = WorkflowEngine(backend=backend)

      async with AcademyWorkflowIntegration(flow) as manager:
            ingest_files_task = manager.create_agent_task(
                  agent_class=FileIngestor,
                  action_name="ingest_files"
            )

            @flow.function_task
            async def ingest_block(files: List[str]):
                  return await ingest_files_task(files)
            
            with add_context_to_log(agent_id=FileIngestor.__name__):
                  response = await ingest_block(files=[
                        "examples/LLM/03-file-upload/utils/files/ny.png",
                        "examples/LLM/03-file-upload/utils/files/lecture_12_26022025.pdf"
                  ])
                  with add_context_to_log(temp="foo"):
                        logger.warning("This is a fake warning")
                  logger.debug(f"Response is: {response}")
                  

      await flow.shutdown()

if __name__ == "__main__":
      asyncio.run(main())