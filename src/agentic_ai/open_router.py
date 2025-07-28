from .utils.core.schemas import LLMResponse, ExtraResponseSettings, ToolSpec
from .utils.core.function_calling import FunctionalToolkit, tool_registry
from agentic_ai.utils import add_context_to_log

import json
import os
import logging
import base64
import asyncio
import aiofiles
import time
import inspect
from typing import Optional, Any, Tuple, List, Dict, Type, Callable, Coroutine
import concurrent.futures


from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEV_INSTRUCTIONS = """
            You must use the provided tools.
            - When a tool is relevant, **immediately call the tool without any conversational filler or "thinking out loud" text.**
            - If you have successfully gathered all necessary information from tool calls to answer the user's request, provide the final answer directly.
            - If something is not sufficiently clear, ask for clarifications.
            - If you are needing a tool but you dont have access to it, you have to sttop and specify that you need it.
            - If there is any logical error in the request, express it, then stop.
            """

class AIAgent:
    """An agent class adapted for the OpenAI SDK, compatible with OpenRouter.

    This class handles system instructions, structured JSON output, and function calling.

    Attributes:
        number_of_interactions (int): Number of loops through the function cycle.
        interactions_limit (int): Usage limit for recursive loops in the function cycle.
        cumulative_token_usage (dict): Cumulative token usage at each loop in the cycle.
        client (AsyncOpenAI): OpenAI async client.
        agent_name (str): Metadata for logging purposes.
        model_name (str): LLM model to be used.
        sys_instruction (str): Extra dev instructions to add to the LLM.
        response_schema (Type[BaseModel]): Pydantic models for structured output from the model's final response.
        settings (dict): Extra settings for the model's completion method (e.g., temperature, max_tokens).
        tools_to_use (dict): Filtered tools to use for the given class instance.
        toolkit (FunctionalToolkit): Instance of FunctionalToolkit with filtered tools.
    """
    number_of_interactions = 0
    interactions_limit = 10 
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
                 tools: Optional[List[str]] = [],
                 extra_response_settings: Optional[Type[ExtraResponseSettings]] = ExtraResponseSettings(),
                 ) -> None:
        """Initializes the agent for use with OpenRouter.

        Args:
            agent_name (str): A name for the agent for logging purposes.
            model_name (str, optional): The LLM model to use. Defaults to "anthropic/claude-sonnet-4".
            sys_instructions (str, optional): The system prompt for the model. Defaults to None.
            response_schema (Type[BaseModel], optional): A Pydantic model to structure the LLM's JSON output. Defaults to None.
            tools (List[str], optional): A list of tool names to use. Defaults to [].
            extra_response_settings (Type[ExtraResponseSettings], optional): Additional parameters for the OpenAI API call (e.g., temperature, max_tokens). Defaults to ExtraResponseSettings().

        Raises:
            ValueError: If the OpenRouter API key is not found in the environment variables.
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
        self.settings = self._set_up_settings(extra_response_settings)
        self.tools_to_use = self._set_up_toolkit(tools=tools) if tools else {}
        self.toolkit = FunctionalToolkit(self.tools_to_use)   
    
    def _set_up_toolkit(self, tools: Optional[List[Callable]] = None) -> dict[str, ToolSpec]:
        """Sets up the toolkit by filtering the global tool registry for the specified tools.

        Args:
            tools (List[str], optional): List of tool names to include. Defaults to None.

        Returns:
            dict[str, ToolSpec]: Dictionary of tool names to ToolSpec objects.
        """
        tools_to_use = {}
        for tool_name in tools:
            if tool_name in tool_registry.keys():
                   tools_to_use[tool_name] = tool_registry[tool_name]
            else:
                  logger.warning(f"Tool '{tool_name}' was requested for agent '{self.agent_name}' but not found in global tool_registry.")
        return tools_to_use
   
    def _set_up_settings(self, extra_response_settings: ExtraResponseSettings) -> Dict:
        """Sets up the settings for the model completion call, including response format if a schema is provided.

        Args:
            extra_response_settings (ExtraResponseSettings): Additional settings for the model completion.

        Returns:
            dict: Dictionary of settings for the model completion call.
        """
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

    def _extract_structure(self, file_extension:str, base_64_string: str, file_path:str) -> Dict:
        """Extracts the structure for file or image input to be sent to the OpenRouter API.

        Args:
            file_extension (str): The file extension (e.g., '.png', '.pdf').
            base_64_string (str): The base64-encoded file content.
            file_path (str): The path to the file.

        Returns:
            dict: The structured dictionary for the file or image.
        """
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
    
    async def _process_single_file(self, file_path: str) -> Dict:
        """Processes a single local file into the OpenRouter API format.

        Args:
            file_path (str): The path to the file.

        Returns:
            List[Dict]: The processed file structure.
        """
        async with aiofiles.open(file_path, "rb") as f:
                with add_context_to_log(file_name=f.name):
                    file_size_bytes = os.path.getsize(file_path)
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    logger.debug(f"File size is: {round(file_size_mb,2)} MB")

                    content = await f.read()
                    base_64_string = base64.b64encode(content).decode("utf-8")
                    file_extension = os.path.splitext(file_path)[1].lower()
                    
                    structure = self._extract_structure(file_extension=file_extension,
                                                         base_64_string=base_64_string,
                                                         file_path=file_path)
                    
                    return structure
                
    async def _process_files(self, files_paths: List[str]) -> List[Dict]:
        """Processes multiple files asynchronously into the OpenRouter API format.

        Args:
            files_paths (List[str]): List of file paths.

        Returns:
            List: List of processed file structures.
        """
        tasks = [self._process_single_file(file_path) for file_path in files_paths] 
        processed_files = await asyncio.gather(*tasks)
        return processed_files
    
    async def _extract_results_tools(
        self,
        messages: List[Dict[str, str]],
        tool_call_info_map: Dict[Any, Tuple[str, int]],
        sync_futures: List[concurrent.futures.Future],
        async_tasks: List[Coroutine[Any, Any, Any]]
    ) -> List[Dict[str, str]]:
        """Extracts results from tool calls, both asynchronous and synchronous, and appends them to the messages list.

        Args:
            messages (List[Dict[str, str]]): The current list of messages.
            tool_call_info_map (Dict[Any, Tuple[str, int]]): Mapping of tasks/futures to (function_name, tool_call_id).
            sync_futures (List[concurrent.futures.Future]): List of synchronous futures.
            async_tasks (List[Coroutine]): List of asynchronous tasks.

        Returns:
            List[Dict[str, str]]: Updated messages list with tool results.
        """
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
    
    async def _complete_tool_calling_cycle(self, response: ChatCompletion, messages: List[dict[str, str]]) -> ChatCompletion:
        """Handles the tool calling cycle: observes if a tool call is requested, executes the tool call, feeds the output back to the model, and recursively continues if needed.

        Args:
            response (ChatCompletion): The current model response.
            messages (List[dict[str, str]]): The conversation context.

        Returns:
            ChatCompletion: The final response object after all tool calls are completed.
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
                        logger.debug(f"Functions in toolkit looks like: {self.toolkit.funcs}")

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
                        
                messages_with_tool_results = await self._extract_results_tools(messages=messages, 
                                                                              tool_call_info_map=tool_call_info_map,
                                                                              sync_futures=sync_futures, 
                                                                              async_tasks=async_tasks)
                response = await self._generate_completition(
                                                        messages=messages_with_tool_results,
                                                        tools=self.toolkit.schematize() if self.toolkit else None,
                                                        )
                self._log_response(response)
                return await self._complete_tool_calling_cycle(response=response, messages=messages)
        else:
            if not under_max_limit_of_interactions_reached:
                logger.warning(f"Exiting tool calling cycle prematurely after reaching {self.number_of_interactions} number of interactions")
            return response
 
    async def _generate_completition(self, messages, tools: Optional[Any] = None) -> ChatCompletion:
        """Generates a model completion using the provided messages and tools.

        Args:
            messages (list): The conversation messages.
            tools (Any, optional): The tools to provide to the model. Defaults to None.

        Returns:
            ChatCompletion: The model's response.
        """
        response = await self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    tools = tools if tools else None,
                    **self.settings
                )
        return response
    
    def _update_cumulative_token_usage(self, response: ChatCompletion) -> None:
        """Updates the cumulative token usage statistics from the model response.

        Args:
            response (ChatCompletion): The model response containing usage information.
        """
        usage = getattr(response, 'usage', None)
        if usage:
            self.cumulative_token_usage['prompt_tokens'] += getattr(usage, 'prompt_tokens', 0)
            self.cumulative_token_usage['completion_tokens'] += getattr(usage, 'completion_tokens', 0)
            self.cumulative_token_usage['total_tokens'] += getattr(usage, 'total_tokens', 0)
    
    def _log_response(self, response: ChatCompletion) -> None:
        """Logs the full response, text response, reasoning, and updates token usage.

        Args:
            response (ChatCompletion): The model response to log.
        """
        logger.debug(f"(📦) Full response: {response}")
        logger.debug(f"(✏️) Text response: {response.choices[0].message.content}")
        self._update_cumulative_token_usage(response)
        reasoning = getattr(response.choices[0].message, 'reasoning', None)
        if reasoning:
            logger.debug(f"(🧠) Reasoning response: {reasoning}")
        else:
            logger.debug("(🧠) No reasoning provided in the message.")
    
    def _summary_log(self, starting_time: int) -> None:
        """Logs a summary of cumulative token usage, number of interactions, and elapsed time.

        Args:
            starting_time (int): The time when the prompt started.
        """
        logger.info(f"(💰) Cumulative token usage: prompt={self.cumulative_token_usage['prompt_tokens']}, completion={self.cumulative_token_usage['completion_tokens']}, total={self.cumulative_token_usage['total_tokens']}")
        logger.info(f"(🛠️) {self.number_of_interactions} interactions occured in function calling")
        if self.number_of_interactions == 0 and self.tools_to_use:
             logger.warning("The LLM hasnt invoked any function/tool, even tho u passed some tool definitions")
        logger.info(f"(⏱️) Took {round(time.time() - starting_time,2)} seconds to fullfill the given prompt")

    async def _generate_completition(self, messages: List[Dict], tools: Optional[Any] = None) -> ChatCompletion:
        """Generates a model completion using the provided messages and tools. (Duplicate method, see above.)

        Args:
            messages (list): The conversation messages.
            tools (Any, optional): The tools to provide to the model. Defaults to None.

        Returns:
            ChatCompletion: The model's response.
        """
        logger.debug(f"Adding the following settings: {self.settings}")
        logger.debug(f"Message is: {messages}")
        response = await self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    tools = tools if tools else None,
                    **self.settings
                )
        return response
        
    def _process_response(self, response: ChatCompletion) -> LLMResponse:
        """Processes the final ChatCompletion object to extract relevant data and log interactions.

        Args:
            response (ChatCompletion): The final model response.

        Returns:
            LLMResponse: The processed response containing the final and parsed responses.
        """
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
                   files_path: Optional[List[str]] = None,
                   n_of_attempts: Optional[int] = 2) -> LLMResponse:
        """Sends a prompt to the LLM, handles multimodal content and function calling, and returns the final response.

        Args:
            message (str): The user prompt to send to the LLM.
            files_path (List[str], optional): List of file paths to include. Defaults to None.
            n_of_attempts (int, optional): Number of attempts to try the prompt. Defaults to 2.

        Returns:
            LLMResponse: The processed response from the LLM.

        Raises:
            Exception: If an error occurs during processing.
        """
        for attempt_number in range(1, n_of_attempts+1):
            try: 
                with add_context_to_log(agent_name=self.agent_name, model_name=self.model_name, prompt_attempt=attempt_number):
                        starting_time = time.time()
                        self.number_of_interactions = 0
                        logger.info(f"Starting prompt. {"Files included" if files_path else "No files included."} with model {self.model_name}")
                        messages = []    
                        user_content = [{"type": "text", "text": message}]
                        if files_path:
                            processed_files = await self._process_files(files_paths=files_path)
                            user_content.extend(processed_files)
                        
                        # Appending context (user message, developer instructions (fixed + user-provided))
                        if self.sys_instructions: 
                            messages.append({"role": "developer", "content": self.sys_instructions})
                        messages.append({"role": "user", "content": user_content})
                        messages.append({"role": "developer", "content": DEV_INSTRUCTIONS})
                        
                        response = await self._generate_completition(
                            messages=messages,
                            tools=self.toolkit.schematize() if self.toolkit else None,
                        )

                        # Logging initial response
                        self._log_response(response=response)

                        # Calling tools if needed
                        if response.choices[0].message.tool_calls:
                            response = await self._complete_tool_calling_cycle(response=response, messages=messages)

                        # Loggin final metrics
                        self._summary_log(starting_time=starting_time)
                        processed_response =  self._process_response(response)
                        logger.debug(f"Final text response is: {processed_response.final_response}")
                        logger.debug(f"Final parsed response is: {processed_response.parsed_response}")
                        return processed_response
            except Exception as e:
                raise e


       
class ToolkitBase():
    """Base class for toolkits. Provides utility to extract tool names from the class.
    """
    def extract_tools_names(self) -> List[str]:
        """Extracts the names of all callable tools defined in the class.

        Returns:
            List[str]: List of tool names.
        """
        tool_names = []
        for name, attr in self.__class__.__dict__.items():
            if callable(attr) and inspect.isfunction(attr) and not name.startswith("_") and name != "extract_tools_names":
               tool_names.append(name)
        logger.debug(f"For {self.__class__} toolkit class, the following tools have been registered: {tool_names}")
        return tool_names

