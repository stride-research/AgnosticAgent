from .schemas import ChunkNotNamed
from agentic_ai.utils import tool
from agentic_ai import ToolkitBase, AIAgent

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ChunkProcessorToolkit(ToolkitBase):

      def activity1():
            ...
      
      def activity2():
            ...

class OrchestratorToolkit(ToolkitBase):
      class ProcessChunkSchema(BaseModel):
            chunk_name: str = Field(..., description="Name of the logical chunk")
            chunk_text: str = Field(..., description="The actual text in the chunk to be processed")

      @tool(schema=ProcessChunkSchema)
      def process_chunk(chunk_name: str, chunk_text: str):
            """Spwans a new subagent. Process a given text from a chunk"""
            chunkProcessor = AIAgent(
                  agent_name=chunk_name,
                  sys_instructions="Given some text return a summary of it and a single keyword",
                  response_schema=ChunkNotNamed,
                  tools=[]
                  )
            response = chunkProcessor.prompt(message=f"This is the chunk of text: {chunk_text}")
            return response
