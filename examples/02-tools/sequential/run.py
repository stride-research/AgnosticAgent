import asyncio
import argparse
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

async def run_example(backend="ollama", model="qwen3:8b"):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openrouter", "openai"])
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args()
    
    asyncio.run(run_example(backend=args.backend, model=args.model))
