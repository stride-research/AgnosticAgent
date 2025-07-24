from __future__ import annotations

import asyncio
import logging
import warnings

from agentic_ai import AIAgent

logger = logging.getLogger(__name__)

warnings.warn("SUPPORT FOR FILES IS NOT YET ENABLED")

files_path = [
                        #"examples/03-file-upload/utils/files/lecture_12_26022025.pdf",
                        "examples/03-file-upload/utils/files/ny.png"
                  ]

def run_example():
      LLMAgent = AIAgent(
                  agent_name="File ingestor",
                  sys_instructions="You have to provide concise explanations of the uploaded files",
                  model_name="google/gemini-2.0-flash-001",
            )

      response = LLMAgent.prompt(message="Describe ALL the uploaded artifacts in less than 10 words for each", 
                                           files_path=files_path)
      
      logger.info(f"Response is {response}")

if __name__ == "__main__":
    run_example()