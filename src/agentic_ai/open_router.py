import json
import os
import logging
import base64
from typing import Optional, Any, Tuple, List, Dict, Type


from .utils.core.function_calling import LLMProcessingLogs
from .utils.core.schemas import LLMResponse, Interaction, InteractionType, StageType, ExtraResponseSettings
from .utils.core.function_calling import FunctionsToToolkit, tool_registry
from agentic_ai.utils import add_context_to_log

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

NUMBER_OF_ITERS = 0

class AIAgent:
    """ 
    An agent class adapted for the OpenAI SDK, compatible with OpenRouter.
    It handles system instructions, structured JSON output, and function calling.

    TODO:
        - Bound interactions in tool cycle
        - Consumption usage logging
    """
    def __init__(self, 
                 agent_name: str,
                 model_name: str = "google/gemini-2.5-pro",
                 sys_instructions: Optional[str] = None, 
                 response_schema: Optional[Type[BaseModel]] = None,
                 extra_response_settings: Optional[Type[ExtraResponseSettings]] = ExtraResponseSettings(),
                 ) -> None:
        """
        Initializes the agent for use with OpenRouter.

        Parameters:
            agent_name: A name for the agent for logging purposes.
            sys_instructions: The system prompt for the model.
            response_schema: A Pydantic model to structure the LLM's JSON output.
            functions_toolkit: A list of Python functions the model can call.
            kwargs: Additional parameters for the OpenAI API call (e.g., temperature, max_tokens).
        """
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("Couldn't find OpenRouter's API key in OPEN_ROUTER_API_KEY environment variable.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.agent_name = agent_name
        self.model_name = model_name
        self.sys_instructions = sys_instructions
        self.LLMProcessingLogs = LLMProcessingLogs(agentName=self.agent_name)
        self.response_schema = response_schema
        self.settings = self.__set_up_settings(extra_response_settings)
        self.toolkit = FunctionsToToolkit(tool_registry)
   
    def __set_up_settings(self, extra_response_settings: ExtraResponseSettings):

        params = extra_response_settings.model_dump(
            exclude_none=True
        )
        if self.response_schema:
           params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_schema.__name__, 
                    "strict": True, 
                    "schema": self.response_schema.model_json_schema() 
                }
            }
        return params

    
    def __process_files(self, files: List[str]) -> List[Dict]:
        """Processes local files into OpenRouter API format."""
        processed_files = []
        for file_path in files:
            with open(file_path, "rb") as f:
                with add_context_to_log(file_name=f.name):
                    file_size_bytes = os.path.getsize(file_path)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    logger.debug(f"File size is: {file_size_mb}MB")

                    base_64_string = base64.b64encode(f.read()).decode("utf-8")
                    
                    # Detect file type and set appropriate content type
                    file_extension = os.path.splitext(file_path)[1].lower()
                    
                    if file_extension in ['.png', '.jpg', '.jpeg', '.webp']:
                        # Handle images
                        content_type = 'image/png' if file_extension == '.png' else 'image/jpeg' if file_extension in ['.jpg', '.jpeg'] else 'image/webp'
                        structure = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{base_64_string}"
                            }
                        }
                    elif file_extension == '.pdf':
                        # Handle PDFs
                        structure = {
                            "type": "file",
                            "file": {
                                "filename": os.path.basename(file_path),
                                "file_data": f"data:application/pdf;base64,{base_64_string}"
                            }
                        }
                    else:
                        # Default to PDF format for unknown types
                        logger.warning(f"Unknown file type {file_extension}, treating as PDF")
                        structure = {
                            "type": "file",
                            "file": {
                                "filename": os.path.basename(file_path),
                                "file_data": f"data:application/pdf;base64,{base_64_string}"
                            }
                        }
                    
                    processed_files.append(structure)
        return processed_files
    def __complete_tool_calling_cycle(self, response: ChatCompletion, messages: List[dict[str, str]]):
        """
        
        Receives a response + message (context), observes if a tool call is requested, executes the given
        tool call, feeds the output back to the model, then recursively calls back this procedure. 
        If no tool call is needed it will return the response object.

        To be done: parallel function call cycle support, upper bound for tool cycle (e.g., exit the recursive loop after max 10 iters)
        
        """
        global NUMBER_OF_ITERS
        messages.append(response.choices[0].message.dict()) # Adding to conversation context the tool call request 

        NUMBER_OF_ITERS += 1 # debugging purposes

        tool_calls = response.choices[0].message.tool_calls
        logger.debug(f"Tool calls: {tool_calls}")
        # if tool_calls:
        #     tool_calls = tool_calls[:1] # TODO: Remove and allow for multiple functions call in parallel
        #     for tool_call in tool_calls:
        #         logger.debug(f"Tool calling: {tool_call}")
        #         function_name = tool_call.function.name
        #         function_args = json.loads(tool_call.function.arguments)
        #         logger.debug(f"Function to call: {function_name}")
        #         logger.debug(f"Function params to call with: {json.dumps(function_args, indent=2)}")
        #         output_result = self.toolkit.execute(func_name=function_name, **function_args)
        #         logger.debug(f"Output result: {output_result}")

        #         messages.append( # Adding to conversation context the output of the function 
        #                 {
        #                     "role": "tool",
        #                     "tool_call_id": tool_call.id,
        #                     "name": function_name,
        #                     "content": str(output_result)
        #                 }
        #         )

        #         logger.debug("SECOND RESPONSE from OpenAI:")
        #         logger.debug("Going for the second completition")
        #         response = self.__generate_completition(messages=messages)
        #         self.__log_response(response)
        #         return self.__complete_tool_calling_cycle(response=response, messages=messages)
        # else:
        #     return response
        return response

    def __process_response(self, response: ChatCompletion) -> LLMResponse:
        """Processes the final ChatCompletion object to extract relevant data and log interactions."""

        prompt_response = response.choices[0].message.content
        
        if self.response_schema:
            json_dict = json.loads(prompt_response)
            parsed_data = self.response_schema.model_validate(json_dict)
        
        return LLMResponse(
            final_response=prompt_response,
            parsed_response=parsed_data if self.response_schema else None
        )
    
    def __log_response(self, response: ChatCompletion):
        print("Raw response from OpenAI:")
        logger.debug(response)
        print("="*30)
        logger.debug(f"Response __dict__: {response.__dict__}")
        print("="*30)
        logger.debug(f"Message dict(): {response.choices[0].message.dict()}")
        print("="*30)
        logger.debug(f"Message: {response.choices[0].message}")
        print("="*30)
        reasoning = getattr(response.choices[0].message, 'reasoning', None)
        if reasoning:
            print(f"Reasoning: {reasoning}")
        else:
            print("No reasoning provided in the message.")
        print("="*30)
    
    def __generate_completition(self, messages, tools: Optional[Any] = None) -> ChatCompletion:
        logger.debug(f"Adding the following settings: {self.settings}")
        logger.debug(f"Message is: {messages}")
        response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    tools = tools if tools else None,
                    **self.settings
                )
        return response
        
    def prompt(self,
                   message: str, 
                   files: Optional[List[str]] = None) -> LLMResponse:
        """
        Sends a prompt to the LLM, handles multimodal content and function calling, and returns the final response.
        """

        messages = []    
        user_content = [{"type": "text", "text": message}]
        if files:
            processed_files = self.__process_files(files)
            user_content.extend(processed_files)
        if self.sys_instructions:
            messages.append({"role": "developer", "content": self.sys_instructions})
        messages.append({"role": "user", "content":user_content})
        messages.append({"role": "developer", "content":"If you think you need to call a function do so immediately in the given request. \
                         If something is not sufficiently clear ask for clarifications."}) # Sometimes the model says it says it will try to call in the next request
        logger.debug(f"Message is: {messages}")
        
        response = self.__generate_completition(
            messages=messages,
            tools=self.toolkit.schematize() if self.toolkit else None,
        )

        #self.__log_response(response=response)

        logger.debug(f"Tool calls: {response.choices[0].message.tool_calls}")
        if response.choices[0].message.tool_calls:
            response = self.__complete_tool_calling_cycle(response=response, messages=messages)
            logger.debug(f"Total number of tools cycles: {NUMBER_OF_ITERS}")

        processed_response =  self.__process_response(response)
        return processed_response

       