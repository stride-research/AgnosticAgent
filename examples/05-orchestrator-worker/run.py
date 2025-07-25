
from .utils.schemas import ORMResponseSchema, ChunkerResponseSchema
from .utils.toolkit import OrchestratorToolkit
from agentic_ai import AIAgent
from agentic_ai.utils import ExtraResponseSettings

from typing import List
import logging
import asyncio

logger = logging.getLogger(__name__)

model_to_use = "google/gemini-2.5-pro"

async def run_example(file_path: str):
      
      # ORM Agent
      # ORMAgent = AIAgent(
      #                   agent_name="ORM",
      #                   model_name=model_to_use,
      #                   sys_instructions="Given a file extract all its text in HTML syntaxis. Extract only the text. Forget about the images. IMPORTANT TO RESPOND IN HTML",
      #                   response_schema=ORMResponseSchema, 
      #                   tools=[]
      #                   )
      
      # ORMResponse = await ORMAgent.prompt(message="Extract all the text from this file. RETURN THE TEXT IN HTML SYNTAXIS", files_path=[file_path])   

      # ORM_extracted_text = ORMResponse.parsed_response.extracted_text

      ORM_extracted_text = """
      <!DOCTYPE html>
      <html>
      <head>
      <title>Letter from IE University</title>
      </head>
      <body>

      <p>ie</p>
      <p>Dr Robert Polding</p>
      <p>Assistant Vice Dean of Computing and Data Science</p>
      <p>School of Science and Technology</p>
      <p>IE University</p>
      <p>P. de la Castellana, 259, Fuencarral-El Pardo, 28046 Madrid</p>
      <p>Dear Javier Domínguez Segura,</p>
      <p>I am writing to congratulate you on an exceptional performance over the last academic year. I want to congratulate you on all the effort you have put in and on performing so well. This shows you have gone above and beyond the requirements for your degree and have obtained an exceptional grade, placing you in the top 10% of students. Congratulations on being among the top achievers in your 2<sup>nd</sup> year of study. This outstanding result is a reflection of your hard work, commitment, and academic excellence. Your performance sets a strong example for others and demonstrates the high standards we value at our institution. We are proud of your achievements and look forward to seeing your continued success in the future.</p>
      <p>Best Regards,</p>
      <p>Dr Polding</p>
      <p>R. Polly</p>

      </body>
      </html>"""

      # Orchestrator Agent 
      agent_name = "Orchestrator"
      OrchestratorAgent = AIAgent(
            agent_name=agent_name,
            model_name=model_to_use,
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
      path = "examples/05-orchestrator-worker/media/Untitled document (1).pdf"
      #path = "examples/05-orchestrator-worker/media/Letter - Javier Domínguez Segura.pdf"
      asyncio.run(run_example(file_path=path))

