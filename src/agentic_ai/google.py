import json
import os
from typing import Optional, Any, Tuple, List, Type
import logging
import base64
import mimetypes
import warnings


from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel

from .utils.core.function_calling import LLMProcessingLogs
from .utils.core.function_calling import Interaction, InteractionType, StageType
from agentic_ai.utils import add_context_to_log

warnings.warn(f"This module is deprecated. Please use OpenRouter version.")

load_dotenv()

logger = logging.getLogger(__name__)

# TODO: Manually manage the function calling cylce in order to allow for true parallelism
# 1. Parse functions 2. Send to LLM 3. Retrieve functions to call + params 4. Invoke in parallel (if needed) 5. Return all results

class ThinkingConfigSchema(BaseModel):
      thinking_budget: int
      include_thoughts: Optional[bool] = False

class InstructionsSchema(BaseModel):
      sys_instructions: Optional[str] = None
      response_schema: Optional[Any] = None
      functions_toolkit: Optional[list] = None
      functions_call_mode: Optional[str] = None
      thinking_config: Optional[ThinkingConfigSchema] = None

class LLMResponse(BaseModel):
      final_response: str
      parsed_response: Optional[str] = None
      response_thoughts: Optional[str] = None

class AIAgent():
      """ 
      Provides simplified set-up of agents.
      Allows to spin-up agents for any context. This class simply sets the proper client for the given llm provider 

      NOTE: Currently harcoded for Gemini's SDK. May seek to generalize for any model later.
      NOTE_2 : For function calling we are using Gemini's Automatic function calling and its native use of parallel function calling and compositional function calling. See more: https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#automatic_function_calling_python_only
      """
      def __init__(self, 
                   agent_name: str,
                   sys_instructions: str="", 
                   thinking_budget: int = 0,
                   include_thoughts: bool = False,
                   response_schema: Optional[BaseModel] = None,
                   functions_toolkit: Optional[list] = None,
                   functions_call_mode: Optional[str] = None,
                   **kwargs
                   ) -> None:
            
            """
            Function call takes on AUTO, ANY, NONE. See more https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#function_calling_modes

            Parameters:
                  kwargs: to be added into types.GenerateContentConfig. Checkout the fill list here: https://ai.google.dev/api/generate-content#generationconfig
            """
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                  raise ValueError("Couldnt find Gemini's API key.")
            self.client = genai.Client(api_key=api_key)
            self.LLMProcessingLogs = LLMProcessingLogs(agentName=agent_name)

            thinking_config = None
            if thinking_budget and thinking_budget != 0:
                  thinking_config = ThinkingConfigSchema(
                        thinking_budget=thinking_budget,
                        include_thoughts=include_thoughts
                  )
            self.instructions = InstructionsSchema(
                  sys_instructions=sys_instructions,
                  response_schema=response_schema,
                  functions_toolkit = functions_toolkit,
                  functions_call_mode = functions_call_mode,
                  thinking_config=thinking_config
            )
            self.__create_settings(self.instructions, **kwargs)

      def __create_settings(self, instructions: InstructionsSchema, **kwargs):
            thinking_config_data = instructions.thinking_config.model_dump() if instructions.thinking_config else None
            gemini_thinking_config = types.ThinkingConfig(**thinking_config_data) if thinking_config_data else None
            gemini_tool_config = types.ToolConfig(
                  function_calling_config=types.FunctionCallingConfig(
                        mode=instructions.functions_call_mode
                  )
            ) if instructions.functions_call_mode else None

            logger.debug(f"THINKING CONFIG: {gemini_thinking_config}")

            self.config = types.GenerateContentConfig(
                  system_instruction=instructions.sys_instructions,
                  thinking_config=gemini_thinking_config,
                  response_mime_type="application/json" if instructions.response_schema else None,
                  response_schema=instructions.response_schema if instructions.response_schema else None,
                  tool_config=gemini_tool_config if instructions.functions_call_mode else None,
                  tools=instructions.functions_toolkit,
                  **kwargs
            )

      def __process_LLM_response(self, response: types.GenerateContentResponse) -> Tuple[Any, str, str]:

            """
            
            Print: 
            - Messaging 
                  - Thinking
                  - Response
            - Processing
                  - Thinking
                  - Tool call
                  - Tool response
            Memory
                  - Final response

            """
            
            response_thoughts = "" # will be empty if didnt set to think in 'instructions'
            final_response = ""
            logger.debug(response)
            for part in response.candidates[0].content.parts: # FINAL ANSWER MESSAGE
                  owner = response.candidates[0].content.role
                  stage=StageType.final
                  if part.text:
                        logger.debug(part)
                        if part.thought:
                              response_thoughts += part.text
                              self.LLMProcessingLogs.push_interaction(Interaction(
                                    stage=stage,
                                    owner=owner,
                                    interaction_type=InteractionType.thought,
                                    interaction_content=part.text
                              ))
                        else:
                              final_response += part.text
                              self.LLMProcessingLogs.push_interaction(Interaction(
                                    stage=stage,
                                    owner=owner,
                                    interaction_type=InteractionType.text,
                                    interaction_content=part.text
                              )
                              )
            for content in response.automatic_function_calling_history: # PROCESSING MESSAGE
                  stage=StageType.processing
                  owner=content.role
                  for part in content.parts:
                        if part.text:
                              if part.thought:
                                    self.LLMProcessingLogs.push_interaction(Interaction(
                                          stage=stage,
                                          owner=content.role,
                                          interaction_type=InteractionType.thought,
                                          interaction_content=part.text
                                    )
                                    )
                              else:
                                    self.LLMProcessingLogs.push_interaction(Interaction(
                                          stage=stage,
                                          owner=content.role,
                                          interaction_type=InteractionType.text,
                                          interaction_content=part.text
                                    )
                                    )
                        if part.function_call:
                              self.LLMProcessingLogs.push_interaction(Interaction(
                                          stage=stage,
                                          owner=content.role,
                                          interaction_type=InteractionType.function_call,
                                          interaction_content=json.dumps(part.function_call.to_json_dict())
                                    )     
                                    )
                        elif part.function_response:
                              self.LLMProcessingLogs.push_interaction(Interaction(
                                          stage=stage,
                                          owner=content.role,
                                          interaction_type=InteractionType.function_response,
                                          interaction_content=json.dumps(part.function_response.to_json_dict())
                                    )
                                    ) 
                              
            self.LLMProcessingLogs.pretty_print()

            logger.debug(f"Response thoughts: {response_thoughts},\
                         instructions_schema: {self.instructions.response_schema}")

            return LLMResponse(
                  final_response=final_response,
                  parsed_response=response.parsed if self.instructions.response_schema else None,
                  response_thoughts=response_thoughts if len(response_thoughts) > 0 else None
            )
      
      @staticmethod
      def __resolve_model_name(model_capabilities:str) -> str:
            models_cognitive_capabilities_map = {
                  "high": "gemini-2.5-pro",
                  "medium": "gemini-2.5-flash",
                  "low": "gemini-2.5-flash-lite-preview-06-17"
            }
            return models_cognitive_capabilities_map[model_capabilities]

      def __process_files(self, files: List[str]):
            files_structured = []
            for file in files:
                  with open(file, "rb") as f:
                        with add_context_to_log(file_name=f.name):
                              file_size_bytes = os.path.getsize(file) 
                              file_size_mb = file_size_bytes / (1024 * 1024)
                              logger.debug(f"File size is: {file_size_mb}MB")
                              mime_type, _ = mimetypes.guess_type(file)
                              if not mime_type:
                                    mime_type = "application/octet-stream"
                              logger.debug(f"File mime type is: {mime_type}")
                              if file_size_mb <= 19:
                                    file_encoded = base64.b64encode(f.read()).decode("utf-8")
                                    structure = types.Part.from_bytes(
                                          data=file_encoded,
                                          mime_type=mime_type
                                    )
                                    files_structured.append(structure)
                              else:
                                    uploaded_file = self.client.files.upload(file=file)
                                    files_structured.append(uploaded_file)
                              logger.info("File processed succesfully")

            return files_structured
                              
      
      def prompt(self, message: str, model_name: str ="gemini-2.5-flash", files: Optional[List[str]]= None) -> LLMResponse:
            """
            Returns response to user, parsed response (if enabled), and thoughts (if enabled)
            """
            contents = [message]

            if model_name in ["high", "medium", "low"]:
                  model_name = self.__resolve_model_name(model_capabilities=model_name)
            
            if files:
                  uploaded_files = self.__process_files(files)
                  logger.info(f"Files processed succesfully")
                  contents.extend(uploaded_files)


            response = self.client.models.generate_content(
                  model=model_name,
                  config=self.config,
                  contents=contents
            )

            return self.__process_LLM_response(response)



            




