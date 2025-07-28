from agentic_ai import CONFIG_DICT
from agentic_ai.utils import add_context_to_log, exception_controller_executor, FunctionalToolkit, tool_registry
from agentic_ai.utils.core.schemas import ExtraResponseSettings, LLMResponse, ToolSpec

import json
import os
import logging
import base64
import asyncio
import aiofiles
import time
import inspect
import aiofiles
import asyncio
from typing import Optional, Any, Tuple, List, Dict, Type, Callable, Coroutine
import concurrent.futures


from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

developer_instructions = """
            You must use the provided tools.
            - When a tool is relevant, **immediately call the tool without any conversational filler or "thinking out loud" text.**
            - If you have successfully gathered all necessary information from tool calls to answer the user's request, provide the final answer directly.
            - If something is not sufficiently clear, ask for clarifications.
            - If you are needing a tool but you dont have access to it, you have to sttop and specify that you need it.
            - If there is any logical error in the request, express it, then stop.
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
                 model_name: str = CONFIG_DICT["AI_agent"]["default_model"],
                 sys_instructions: Optional[str] = None, 
                 response_schema: Optional[Type[BaseModel]] = None,
                 tools: Optional[List[str]] = [],
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
        self.response_schema = response_schema
        self.settings = self.__set_up_settings(extra_response_settings)
        self.tools_to_use = self.__set_up_toolkit(tools=tools) if tools else {}
        self.toolkit = FunctionalToolkit(self.tools_to_use)   
    
    def __set_up_toolkit(self, tools: Optional[List[Callable]] = None) -> dict[str, ToolSpec]:
        tools_to_use = {}
        for tool_name in tools:
            if tool_name in tool_registry.keys():
                   tools_to_use[tool_name] = tool_registry[tool_name]
            else:
                  logger.warning(f"Tool '{tool_name}' was requested for agent '{self.agent_name}' but not found in global tool_registry.")
        return tools_to_use
   
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

    def __extract_structure(self, file_extension:str, base_64_string: str, file_path:str) -> dict:
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
    
    async def __process_single_file(self, file_path: str) -> List[Dict]:
        """Processes local files into OpenRouter API format."""
        async with aiofiles.open(file_path, "rb") as f:
                with add_context_to_log(file_name=f.name):
                    file_size_bytes = os.path.getsize(file_path)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    logger.debug(f"File size is: {round(file_size_mb,2)} MB")

                    content = await f.read()
                    base_64_string = base64.b64encode(content).decode("utf-8")
                    file_extension = os.path.splitext(file_path)[1].lower()
                    
                    structure = self.__extract_structure(file_extension=file_extension,
                                                         base_64_string=base_64_string,
                                                         file_path=file_path)
                    
                    return structure
                
    async def __process_files(self, files_paths: List[str]):
        tasks = [self.__process_single_file(file_path) for file_path in files_paths] 
        processed_files = await asyncio.gather(*tasks)
        return processed_files
    
    async def extract_results_tools(
        self,
        messages: List[Dict[str, str]],
        tool_call_info_map: Dict[Any, Tuple[str, int]],
        sync_futures: List[concurrent.futures.Future],
        async_tasks: List[Coroutine[Any, Any, Any]]
    ) -> List[Dict[str, str]]:
        
        if async_tasks:
                    completed_tasks = await asyncio.gather(*async_tasks, return_exceptions=True)
                    for i, result in enumerate(completed_tasks):
                        original_task = async_tasks[i] # Get the original task
                        function_name_completed, original_tool_call_id = tool_call_info_map[original_task]
                        
                        if isinstance(result, Exception):
                            logger.error(f"(🔧) Async tool call {function_name_completed} failed: {result}")
                            content = f"Error: {str(result)}"
                            raise result
                        else:
                            logger.debug(f"(🔧) Completed async task: {function_name_completed} with data: {result}")
                            content = str(result)
                        
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": original_tool_call_id,
                                "name": function_name_completed,
                                "content": content
                            }
                        )
        if sync_futures:
                    for future in concurrent.futures.as_completed(sync_futures):
                        function_name_completed, original_tool_call_id = tool_call_info_map[future]
                        try:
                            output = future.result()
                            logger.debug(f"(🔧) Completed sync future: {function_name_completed} with output: {output}")
                            content = str(output)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": original_tool_call_id,
                                "name": function_name_completed,
                                "output": json.dumps(output) if isinstance(output, (dict, list)) else str(output),
                            })
                        except Exception as exc:
                            logger.error(f"(🔧) Sync tool call {function_name_completed} failed: {exc}")
                            raise exc
                            content = f"Error: {str(exc)}"
                            # TODO: Add message appending when there is an error 

                      
        return messages
    
    async def __complete_tool_calling_cycle(self, response: ChatCompletion, messages: List[dict[str, str]]):
        """
        
        Receives a response + message (context), observes if a tool call is requested, executes the given
        tool call, feeds the output back to the model, then recursively calls back this procedure. 
        If no tool call is needed it will return the response object.
        
        """
        messages.append(response.choices[0].message.dict()) # Adding to conversation context the tool call request . NOTE: All this much context should probably not be included. To be researched

        tool_calls = response.choices[0].message.tool_calls
        logger.debug(f"(🔧) Tool calls ({len(tool_calls) if tool_calls else 0} tools requested): {tool_calls}")

        under_max_limit_of_interactions_reached = self.number_of_interactions < self.interactions_limit
        if tool_calls and under_max_limit_of_interactions_reached:
            self.number_of_interactions += 1

            with add_context_to_log(interacion_number=self.number_of_interactions):
                sync_futures = []
                async_tasks = []
                tool_call_info_map = {}
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                        
                        procedure = self.toolkit.tools.get(function_name)
                        if not procedure:
                            logger.warning(f"Tool '{function_name}' requested by LLM but not found in toolkit.")
                            messages.append({
                                "tool_call_id": tool_call_id,
                                "output": f"Error: Tool '{function_name}' not found.",
                                "name": function_name
                            })
                            continue

                        executable_method = procedure.get_executable()
                        if procedure.is_coroutine:
                            task = asyncio.create_task(executable_method(**function_args))
                            async_tasks.append(task)
                            tool_call_info_map[task] = (function_name, tool_call_id)
                        else:
                            future = executor.submit(executable_method, **function_args)
                            sync_futures.append(future)
                            tool_call_info_map[future] = (function_name, tool_call.id)
                        
                messages_with_tool_results = await self.extract_results_tools(messages=messages, 
                                                                              tool_call_info_map=tool_call_info_map,
                                                                              sync_futures=sync_futures, 
                                                                              async_tasks=async_tasks)
                response = await self.__generate_completition(
                                                        messages=messages_with_tool_results,
                                                        tools=self.toolkit.schematize() if self.toolkit else None,
                                                        )
                self.__log_response(response)
                return await self.__complete_tool_calling_cycle(response=response, messages=messages)
        else:
            if not under_max_limit_of_interactions_reached:
                logger.warning(f"Exiting tool calling cycle prematurely after reaching {self.number_of_interactions} number of interactions")
            return response
 
    async def __generate_completition(self, messages, tools: Optional[Any] = None) -> ChatCompletion:
        response = await self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    tools = tools if tools else None,
                    **self.settings
                )
        return response
    
    def __update_cumulative_token_usage(self, response: ChatCompletion):
        usage = getattr(response, 'usage', None)
        if usage:
            self.cumulative_token_usage['prompt_tokens'] += getattr(usage, 'prompt_tokens', 0)
            self.cumulative_token_usage['completion_tokens'] += getattr(usage, 'completion_tokens', 0)
            self.cumulative_token_usage['total_tokens'] += getattr(usage, 'total_tokens', 0)
    
    def __log_response(self, response: ChatCompletion):
        logger.debug(f"(📦) Full response: {response}")
        logger.debug(f"(✏️) Text response: {response.choices[0].message.content}")
        self.__update_cumulative_token_usage(response)
        reasoning = getattr(response.choices[0].message, 'reasoning', None)
        if reasoning:
            logger.debug(f"(🧠) Reasoning response: {reasoning}")
        else:
            logger.debug("(🧠) No reasoning provided in the message.")
    
    def __summary_log(self, starting_time: int):
        logger.info(f"(💰) Cumulative token usage: prompt={self.cumulative_token_usage['prompt_tokens']}, completion={self.cumulative_token_usage['completion_tokens']}, total={self.cumulative_token_usage['total_tokens']}")
        logger.info(f"(🛠️) {self.number_of_interactions} interactions occured in function calling")
        if self.number_of_interactions == 0 and self.tools_to_use:
             logger.warning(f"The LLM hasnt invoked any function/tool, even tho u passed some tool definitions")
        logger.info(f"(⏱️) Took {round(time.time() - starting_time,2)} seconds to fullfill the given prompt")

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
        
    def __process_response(self, response: ChatCompletion) -> LLMResponse:
        """Processes the final ChatCompletion object to extract relevant data and log interactions."""

        logger.debug(f"Response is: {response}")

        prompt_response = response.choices[0].message.content
        
        if self.response_schema:
            json_dict = json.loads(prompt_response)
            parsed_data = self.response_schema.model_validate(json_dict)
        
        return LLMResponse(
            final_response=prompt_response,
            parsed_response=parsed_data if self.response_schema else None
        )

    async def prompt(self,
                   message: str, 
                   files_path: Optional[List[str]] = None) -> LLMResponse:
        """
        Sends a prompt to the LLM, handles multimodal content and function calling, and returns the final response.
        """
        async def get_model_response(
                   message: str, 
                   files_path: Optional[List[str]] = None) -> LLMResponse:
            with add_context_to_log(agent_name=self.agent_name, model_name=self.model_name):
                starting_time = time.time()
                self.number_of_interactions = 0
                logger.info(f"Starting prompt. {"Files included" if files_path else "No files included."} with model {self.model_name}")
                messages = []    
                user_content = [{"type": "text", "text": message}]
                if files_path:
                    processed_files = await self.__process_files(files_paths=files_path)
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

                # Logging initial response
                self.__log_response(response=response)

                # Calling tools if needed
                if response.choices[0].message.tool_calls:
                    response = await self.__complete_tool_calling_cycle(response=response, messages=messages)

                # Loggin final metrics
                self.__summary_log(starting_time=starting_time)
                processed_response =  self.__process_response(response)
                logger.debug(f"Final text response is: {processed_response.final_response}")
                logger.debug(f"Final parsed response is: {processed_response.parsed_response}")
                return processed_response
            
        output = await exception_controller_executor.execute_with_retries(func=get_model_response,
                                                                          message=message,
                                                                          files_path=files_path)
        return output


       
class ToolkitBase():
    def extract_tools_names(self):
        tool_names = []
        for name, attr in self.__class__.__dict__.items():
            if callable(attr) and inspect.isfunction(attr) and not name.startswith("_") and name != "extract_tools_names":
               tool_names.append(name)
        logger.debug(f"For {self.__class__} toolkit class, the following tools have been registered: {tool_names}")
        return tool_names

