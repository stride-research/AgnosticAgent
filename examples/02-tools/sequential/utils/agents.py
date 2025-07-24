



from academy.agent import Agent, action
from agentic_ai.LLM_agent import AIAgent, ExtraResponseSettings
from . import toolkit # Load them to the toolkit

from pydantic import BaseModel

import logging

logger = logging.getLogger(__name__)

class Schema(BaseModel):
      math_result: int | float

class Mathematician(Agent):
      thoughts: list

      async def agent_on_startup(self):
            self.thoughts = []
            self.x = 4
            self.y = 3
            self.AIAgent = AIAgent(
                              agent_name="Mathematician",
                              model_name="google/gemini-2.5-pro",
                              sys_instructions="Do some basic arithmetic with the provided tools",
                              response_schema=Schema,
                              extra_response_settings=ExtraResponseSettings(max_tokens=3000)
                        )
      @action
      async def basic_math(self):
            """
            Given two numbers ('a', 'b') it first adds the two ('c'), then multiply 'c' by 'a'.
            """
            message = f"Add {self.x} to {self.y}. Then multiply the result of this operation to {self.x}, then stop"
            response_to_user = self.AIAgent.prompt(message=message)
            logger.debug(f"Response is: {response_to_user}")
            return response_to_user
      
      async def agent_on_shutdown(self) -> None:
            print(f"ALL THOUGHTS HERE: {self.thoughts}")


