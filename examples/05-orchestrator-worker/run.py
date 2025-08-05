
import argparse
from .utils.schemas import ORMResponseSchema, ChunkerResponseSchema
from .utils.toolkit import OrchestratorToolkit
from agentic_ai import LLMAgent
from agentic_ai.utils import ExtraResponseSettings

from ..config import inline_args

from typing import List
import logging
import asyncio

logger = logging.getLogger(__name__)

async def run_example(file_path: str, backend: str, model: str):

      # ORM Agent
      ORMAgent = LLMAgent(
                        llm_backend=backend,
                        agent_name="ORM",
                        model_name=model,
                        sys_instructions="Given a file extract all its text in HTML syntaxis. Extract only the text. Forget about the images. IMPORTANT TO RESPOND IN HTML",
                        response_schema=ORMResponseSchema,
                        tools=[]
                        )

      ORMResponse = await ORMAgent.prompt(message="Extract all the text from this file. RETURN THE TEXT IN HTML SYNTAXIS", files_path=[file_path])

      ORM_extracted_text = ORMResponse.parsed_response.extracted_text

      # Orchestrator Agent
      agent_name = "Orchestrator"
      OrchestratorAgent = LLMAgent(
            llm_backend=backend,
            agent_name=agent_name,
            model_name=model,
            sys_instructions="Given some text input find sections of the text it can logically be chunked into. After that assign each chunk of text to a subagent in order for it to process it and wait until it returns the processed chunk.\
                  You will be provided with some tools in order to spawn as many subagents as needed.\
                  Under no circusmtance should you try to summarize the chunks by yourself. You need to delegate to subagents created with\
                  the provided tools.\
                  Provide a summary of what you did at the end",
            response_schema=ChunkerResponseSchema,
            tools=OrchestratorToolkit().extract_tools_names()
      )

      OrchestratorResponse = await OrchestratorAgent.prompt(
            message=f"This is the text: {ORM_extracted_text}. Spawn subagents for a given section pass the given text and chunk name and wait for the processing of it."
      )

if __name__ == "__main__":
    
    # Ollama can't handle pdfs
    #path = "examples/05-orchestrator-worker/media/Untitled document (1).pdf"
    path = "examples/05-orchestrator-worker/media/ny.png"
    #path = "examples/05-orchestrator-worker/media/Letter - Javier Domínguez Segura.pdf"
    asyncio.run(run_example(file_path=path, backend=inline_args.backend, model=inline_args.model))
