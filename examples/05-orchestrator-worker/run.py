
from .utils.schemas import ORMResponseSchema, ChunkerResponseSchema
from .utils.toolkit import OrchestratorToolkit
from agentic_ai import AIAgent

from typing import List
import logging

logger = logging.getLogger(__name__)

model_to_use = "google/gemini-2.5-flash-lite"

def run_example(file_path: str):
      
      # ORM Agent
      ORMAgent = AIAgent(
                        agent_name="ORM",
                        model_name=model_to_use,
                        sys_instructions="Given a file extract all its text in HTML syntaxis. Extract only the text. Forget about the images. IMPORTANT TO RESPOND IN HTML",
                        response_schema=ORMResponseSchema, 
                        tools=[]
                        )
      
      ORMResponse = ORMAgent.prompt(message="Extract all the text from this file in HTML. RETURN IN HTML SYNTAXIS", files_path=[file_path])   
      
      ORM_extracted_text = ORMResponse.parsed_response.extracted_text

      # Orchestrator Agent 
      OrchestratorAgent = AIAgent(
            agent_name="Orchestrator",
            model_name=model_to_use,
            sys_instructions="Given some text input in HTML find sections of the text it can logically be chunked into. After that assign each chunk of text to a subagent in order for it to process it and wait until it returns the processed chunk.\
                  You will be provided with some tools in order to spawn as many subagents as needed.",
            response_schema=ChunkerResponseSchema,
            tools=OrchestratorToolkit().extract_tools_names()
            
      )

      OrchestratorResponse = OrchestratorAgent.prompt(
            message=f"This is the text: {ORM_extracted_text}. Spawn subagents for a given section pass the given text and chunk name and wait for the processing of it."
      )


if __name__ == "__main__":
      path = "examples/05-orchestrator-worker/media/Untitled document (1).pdf"
      run_example(file_path=path)

