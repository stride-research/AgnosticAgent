from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel


from academy.manager import Manager

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import ThreadExecutionBackend

# Import our integration layer (assuming it's in the same directory)
from agentic_ai import AcademyWorkflowIntegration
from agentic_ai.utils import add_context_to_log
from .utils.agents import WordGuesser, CORRECT_WORD, NUMBER_OF_ITERS

logger = logging.getLogger(__name__)

async def main() -> int:
    """
    Fixed Academy loop example that properly handles loops.
    """
    
    # Create workflow engine
    backend = ThreadExecutionBackend({})
    backend.set_main_loop()
    flow = WorkflowEngine(backend=backend)
    
    # Create integration layer
    async with AcademyWorkflowIntegration(flow) as manager:
        with add_context_to_log(agent_id=WordGuesser.__name__):
            guess_word_task = manager.create_agent_task(
                WordGuesser, "guess_word"
            )

            @flow.function_task
            async def guess_word_block() -> str:
                return await guess_word_task()
            
            for i in range(NUMBER_OF_ITERS):
                selected_word = await guess_word_block()
                print(f"Selected word is: {selected_word}")
                if selected_word.strip().upper() == CORRECT_WORD:
                    print("Word choice was correct")
                    break
                else:
                    print("Word choice was incorrect")

    await flow.shutdown()
    return 0

if __name__ == "__main__":
    asyncio.run(main())
