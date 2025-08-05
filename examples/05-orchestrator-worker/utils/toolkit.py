from .schemas import ChunkNotNamed
from agnostic_agent.utils import tool, add_context_to_log
from agnostic_agent import ToolkitBase, LLMAgent

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
class OrchestratorToolkit(ToolkitBase):
      class ProcessChunkSchema(BaseModel):
            chunk_name: str = Field(..., description="Name of the logical chunk")
            chunk_text: str = Field(..., description="The actual text in the chunk to be processed")

      @tool(schema=ProcessChunkSchema)
      async def process_chunk(chunk_name: str, chunk_text: str):
            """Process a given text from a chunk"""
            logger.debug("Im inside 'proces_chunk'")
            chunkProcessor = LLMAgent(
                  llm_backend="OpenRouter",
                  agent_name=chunk_name,
                  sys_instructions="Given some text return a summary of it and a single keyword",
                  response_schema=ChunkNotNamed,
                  tools=[]
                  )
            response = await chunkProcessor.prompt(message=f"This is the chunk of text: {chunk_text}")
            logger.debug("I can leave 'process_chunk' now")
            return response