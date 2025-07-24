
from typing import List

from academy.agent import Agent, action
from agentic_ai.LLM_agent import AIAgent, ResponseSettings


import logging

logger = logging.getLogger(__name__)

class FileIngestor(Agent):
      async def agent_on_startup(self) -> None:
            self.AIAgent = AIAgent(
                  agent_name=__class__.__name__,
                  sys_instructions="You have to provide concise explanations of the uploaded files",
                  model_name="google/gemini-2.0-flash-001",
            )
      
      @action
      async def ingest_files(self, files: List[str]):
            response = self.AIAgent.prompt(message="Describe ALL the uploaded artifacts in less than 10 words for each", 
                                           files=files)
            return response
