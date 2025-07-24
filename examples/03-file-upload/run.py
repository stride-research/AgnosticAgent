from agentic_ai import AIAgent

import asyncio
import logging
import warnings


logger = logging.getLogger(__name__)


files_path = [
                        "examples/03-file-upload/utils/files/1ST_LAB_SESSION (1).pdf",
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
      
      logger.info(f"FINAL RESPONSE is {response}")

if __name__ == "__main__":
    run_example()