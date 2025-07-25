from .utils.core.function_calling import LLMProcessingLogs
from .utils.core.schemas import LLMResponse, Interaction, InteractionType, StageType, ExtraResponseSettings
from .utils.core.function_calling import FunctionsToToolkit, tool_registry
from agentic_ai.utils import add_context_to_log

import json
import os
import logging
import base64
import asyncio
import aiofiles
from typing import Optional, Any, Tuple, List, Dict, Type

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

developer_instructions = """Given a dish, you need to first provide the ingredients required, then return the price of these ingredients.
            You must use the provided tools.
            - When a tool is relevant, **immediately call the tool without any conversational filler or "thinking out loud" text.**
            - If you have successfully gathered all necessary information from tool calls to answer the user's request, provide the final answer directly.
            - If something is not sufficiently clear, ask for clarifications.
            - If you are needing a tool but you dont have access to it, you have to sttop and specify that you need it.
            """

class AIAgent:
    """
    An agent class adapted for the OpenAI SDK, compatible with OpenRouter.
    It handles system instructions, structured JSON output, and function calling.

    TODO:
        - Consumption usage logging
    """
    number_of_interactions = 0
    interactions_limit = 10 # Usage limit for recursive tool calling
    token_usage = None
    cumulative_token_usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }

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
        
        self.client = AsyncOpenAI(
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

    async def __process_files(self, files: List[str]) -> List[Dict]:
        """Processes local files into OpenRouter API format."""
        processed_files = []
        
        async def process_single_file(file_path: str) -> Dict:
            async with aiofiles.open(file_path, "rb") as f:
                with add_context_to_log(file_name=file_path):
                    file_size_bytes = os.path.getsize(file_path)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    logger.debug(f"File size is: {file_size_mb}MB")

                    content = await f.read()
                    base_64_string = base64.b64encode(content).decode("utf-8")
                    
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
                    
                    return structure
        
        # Process files concurrently
        tasks = [process_single_file(file_path) for file_path in files]
        processed_files = await asyncio.gather(*tasks)
        return processed_files

    async def __complete_tool_calling_cycle(self, response: ChatCompletion, messages: List[dict[str, str]]):
        """
        
        Receives a response + message (context), observes if a tool call is requested, executes the given
        tool call, feeds the output back to the model, then recursively calls back this procedure. 
        If no tool call is needed it will return the response object.
        
        """
        messages.append(response.choices[0].message.dict()) # Adding to conversation context the tool call request . NOTE: All this much context should probably not be included

        tool_calls = response.choices[0].message.tool_calls
        logger.debug(f"Tool calls: {tool_calls}")
        if tool_calls and self.number_of_interactions < self.interactions_limit:
            self.number_of_interactions += 1
            with add_context_to_log(interacion_number=self.number_of_interactions):
                # Use asyncio.gather for concurrent execution of tool calls
                async def execute_tool_call(tool_call):
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    logger.debug(f"Gotta call {function_name} with {function_args}")
                    
                    # Execute the tool call
                    data = await asyncio.to_thread(self.toolkit.execute, function_name, **function_args)
                    logger.debug(f"Completed tool call: {function_name} with data: {data}")
                    
                    return {
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": str(data)
                    }
                
                # Execute all tool calls concurrently
                tool_results = await asyncio.gather(*[execute_tool_call(tc) for tc in tool_calls])
                
                # Add all results to messages
                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "name": result["name"],
                        "content": result["content"]
                    })
                
                response = await self.__generate_completition(
                    messages=messages,
                    tools=self.toolkit.schematize() if self.toolkit else None,
                )
                self.__log_response(response)
                return await self.__complete_tool_calling_cycle(response=response, messages=messages)
        else:
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

    def __update_cumulative_token_usage(self, response: ChatCompletion):
        usage = getattr(response, 'usage', None)
        if usage:
            self.cumulative_token_usage['prompt_tokens'] += getattr(usage, 'prompt_tokens', 0)
            self.cumulative_token_usage['completion_tokens'] += getattr(usage, 'completion_tokens', 0)
            self.cumulative_token_usage['total_tokens'] += getattr(usage, 'total_tokens', 0)

    def __log_response(self, response: ChatCompletion):
        logger.debug(f"Full response: {response}")
        logger.debug(f"Text response: {response.choices[0].message.content}")
        self.__update_cumulative_token_usage(response)
        reasoning = getattr(response.choices[0].message, 'reasoning', None)
        if reasoning:
            print(f"Reasoning response: {reasoning}")
        else:
            print("No reasoning provided in the message.")

    def __summary_log(self):
        logger.info(f"Cumulative token usage: prompt={self.cumulative_token_usage['prompt_tokens']}, completion={self.cumulative_token_usage['completion_tokens']}, total={self.cumulative_token_usage['total_tokens']}")
        logger.info(f"{self.number_of_interactions} interactions occured in function calling")
        if self.number_of_interactions == 0 and tool_registry:
             logger.warning(f"The LLM hasnt invoked any function/tool, even tho u passed some tool definitions")

    async def __generate_completition(self, messages, tools: Optional[Any] = None) -> ChatCompletion:
        logger.debug(f"Adding the following settings: {self.settings}")
        logger.debug(f"Message is: {messages}")
        response = await self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    tools = tools if tools else None,
                    **self.settings
                )
        return response
        
    async def prompt(self,
                   message: str, 
                   files_path: Optional[List[str]] = None) -> LLMResponse:
        """
        Sends a prompt to the LLM, handles multimodal content and function calling, and returns the final response.
        """
        with add_context_to_log(model_name=self.model_name):
            self.number_of_interactions = 0
            logger.info(f"Starting prompt. {"Files included" if files_path else "No files included."} with model {self.model_name}")
            messages = []    
            user_content = [{"type": "text", "text": message}]
            if files_path:
                processed_files = await self.__process_files(files_path)
                user_content.extend(processed_files)
            
            # Appending context (user message, developer instructions (fixed + user-provided))
            if self.sys_instructions: 
                messages.append({"role": "developer", "content": self.sys_instructions})
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "developer", "content": developer_instructions})
            
            response = await self.__generate_completition(
                messages=messages,
                tools=self.toolkit.schematize() if self.toolkit else None,
            )

            # Calling tools if needed
            if response.choices[0].message.tool_calls:
                response = await self.__complete_tool_calling_cycle(response=response, messages=messages)
            
            processed_response = self.__process_response(response)
            self.__summary_log()
            return processed_response