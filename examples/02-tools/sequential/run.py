import asyncio
from .utils import toolkit
from agentic_ai import AIAgent
from agentic_ai.utils import ExtraResponseSettings

import logging

from pydantic import BaseModel


logger = logging.getLogger(__name__)

x = 4
y = 3

class Schema(BaseModel):
      math_result: int | float

async def run_example():
      LLMAgent = AIAgent(
                              agent_name="Mathematician",
                              model_name="google/gemini-2.5-pro",
                              sys_instructions="Do some basic arithmetic with the provided tools",
                              response_schema=Schema
                        )

      message = f"Add {x} to {y}. Then multiply the result of this operation to {x}, then stop"
      response = await LLMAgent.prompt(message=message)

if __name__ == "__main__":
    asyncio.run(run_example())
