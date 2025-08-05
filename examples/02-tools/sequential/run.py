import argparse
from .utils.toolkit import MathematicianToolkit
from agnostic_agent import LLMAgent
from agnostic_agent.utils import ExtraResponseSettings

import logging
import asyncio

from ...config import inline_args


from pydantic import BaseModel


logger = logging.getLogger(__name__)

x = 4
y = 3

class Schema(BaseModel):
      math_result: int | float

async def run_example(backend:str, model:str):
      agent = LLMAgent(
                        llm_backend=backend,
                        agent_name="Mathematician",
                        model_name=model,
                        sys_instructions="Do some basic arithmetic with the provided tools",
                        response_schema=Schema,
                        tools=MathematicianToolkit().extract_tools_names()
                        )

      message = f"Add {x} to {y}. Then multiply the result of this operation to {x}, then stop"
      response = await agent.prompt(message=message)

if __name__ == "__main__":
    asyncio.run(run_example(backend=inline_args.backend, 
                            model=inline_args.model))
