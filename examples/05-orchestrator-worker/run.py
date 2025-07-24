
from .utils.schemas import ORMResponseSchema, ChunkerResponseSchema
from agentic_ai import AIAgent

from typing import List
import logging

logger = logging.getLogger(__name__)

def run_example(file_path: str):
      
      # ORM Agent
      ORMAgent = AIAgent(
                        agent_name="ORM",
                        sys_instructions="Given a file extract all its text in HTML syntaxis. Extract only the text. Forget about the images. IMPORTANT TO RESPOND IN HTML",
                        response_schema=ORMResponseSchema
                        )
      
      ORMResponse = ORMAgent.prompt(message="Extract all the text from this file in HTML.", files_path=[file_path])   

      ORM_extracted_text = ORMResponse.parsed_response.extracted_text

      OrchestratorAgent = AIAgent(
            agent_name="Orchestrator",
            sys_instructions="Given some text input in HTML find sections of the text it can logically be chunked into. After that assign each chunk of text to a subagent in order for it to process it and wait until it returns the processed chunk.",
            response_schema=ChunkerResponseSchema
      )

      OrchestratorResponse = OrchestratorAgent.prompt(
            message=f"This is the text: {ORM_extracted_text} please provide the given logical chunks"
      )


if __name__ == "__main__":
      path = "examples/05-orchestrator-worker/media/Letter - Javier Domínguez Segura.pdf"
      run_example(file_path=path)

