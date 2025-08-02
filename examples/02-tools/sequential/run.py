import asyncio
from .utils.toolkit import MathematicianToolkit
from agentic_ai import LLMAgent
from agentic_ai.utils import ExtraResponseSettings

import logging
import asyncio

from pydantic import BaseModel


logger = logging.getLogger(__name__)

x = 4
y = 3

class Schema(BaseModel):
      math_result: int | float

async def run_example():
      agent = LLMAgent(
                        llm_backend="OpenRouter",
                        agent_name="Mathematician",
                        model_name="google/gemini-2.5-pro",
                        sys_instructions="Do some basic arithmetic with the provided tools",
                        response_schema=Schema,
                        tools=MathematicianToolkit().extract_tools_names()
                        )

      message = f"Add {x} to {y}. Then multiply the result of this operation to {x}, then stop"
      response = await agent.prompt(message=message)

if __name__ == "__main__":
    asyncio.run(run_example())
