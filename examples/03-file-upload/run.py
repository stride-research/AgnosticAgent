from agentic_ai import LLMAgent

import asyncio
import logging
import warnings


logger = logging.getLogger(__name__)


files_path = [
                        "examples/03-file-upload/utils/files/1ST_LAB_SESSION (1).pdf", # Ollama can't handle PDFs
                        "examples/03-file-upload/utils/files/ny.png"
                  ]

async def run_example():
      agent = LLMAgent(
                  llm_backend="ollama",
                  agent_name="File ingestor",
                  sys_instructions="You have to provide concise explanations of the uploaded files",
                  model_name="gemma3n:latest",
                  tools=[]
            )

      response = await agent.prompt(message="Describe ALL the uploaded artifacts in less than 10 words for each",
                                           files_path=files_path)

      logger.info(f"FINAL RESPONSE is {response}")

if __name__ == "__main__":
    asyncio.run(run_example())
