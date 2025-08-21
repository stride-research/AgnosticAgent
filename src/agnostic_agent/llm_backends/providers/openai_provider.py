
"""
TO BE DONE:
    - Break class into: Configuration manager, File processing, Tool execution, Logging 
"""

import asyncio
import base64
import concurrent.futures
import json
import logging
import os
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Type

import aiofiles
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from agnostic_agent.utils import add_context_to_log

from ...utils.core.function_calling.openai import FunctionalToolkit, tool_registry
from ...utils.core.schemas import ExtraResponseSettings, LLMResponse, ToolSpec
from .base_llm_provider import BaseLLMProvider

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

class OpenAIProvider(BaseLLMProvider):
    def __init__(self,
                agent_name: str,
                model_name: str,
                api_key: str,
                base_url: str,
                sys_instructions: Optional[str] = None,
                response_schema: Optional[Type[BaseModel]] = None,
                tools: Optional[List[str]] = [],
                extra_response_settings: Optional[Type[ExtraResponseSettings]] = ExtraResponseSettings(),
                ) -> None:
        """Initializes the OpenAI-compatible provider.

        Args:
            agent_name (str): A name for the agent for logging purposes.
            model_name (str): The LLM model to use.
            api_key (str): The API key for authentication.
            base_url (str): The base URL for the API endpoint.
            sys_instructions (str, optional): The system prompt for the model. Defaults to None.
            response_schema (Type[BaseModel], optional): A Pydantic model to structure the LLM's JSON output. Defaults to None.
            tools (List[str], optional): A list of tool names to use. Defaults to [].
            extra_response_settings (Type[ExtraResponseSettings], optional): Additional parameters for the OpenAI API call. Defaults to ExtraResponseSettings().
        """
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.agent_name = agent_name
        self.model_name = model_name
        self.sys_instructions = sys_instructions
        self.response_schema = response_schema
        self.tools = tools
        self.settings = self._set_up_settings(extra_response_settings)
        self.tools_to_use = self._set_up_toolkit(tools=tools) if tools else {}
        self.toolkit = FunctionalToolkit(self.tools_to_use)

    def _set_up_toolkit(self, tools: Optional[List[Callable]] = None) -> dict[str, ToolSpec]:
        """Sets up the toolkit by filtering the global tool registry for the specified tools."""
        tools_to_use = {}
        for tool_name in tools:
            if tool_name in tool_registry.keys():
                tools_to_use[tool_name] = tool_registry[tool_name]
            else:
                logger.warning(f"Tool '{tool_name}' was requested for agent '{self.agent_name}' but not found in global tool_registry.")
        return tools_to_use

    def _set_up_settings(self, extra_response_settings: ExtraResponseSettings) -> Dict:
        """Sets up the settings for the model completion call, including response format if a schema is provided."""
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
        """Extracts the structure for file or image input to be sent to the API."""
        if file_extension in ['.png', '.jpg', '.jpeg', '.webp']:
            content_type = 'image/png' if file_extension == '.png' else 'image/jpeg' if file_extension in ['.jpg', '.jpeg'] else 'image/webp'
            structure = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content_type};base64,{base_64_string}"
                }
            }
        elif file_extension == '.pdf':
            structure = {
                "type": "file",
                "file": {
                    "filename": os.path.basename(file_path),
                    "file_data": f"data:application/pdf;base64,{base_64_string}"
                }
            }
        else:
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
        """Processes a single local file into the API format."""
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
        """Processes multiple files asynchronously into the API format."""
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
        """Extracts results from tool calls and appends them to the messages list."""
        if async_tasks:
            completed_tasks = await asyncio.gather(*async_tasks, return_exceptions=True)
            for i, result in enumerate(completed_tasks):
                original_task = async_tasks[i]
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
                        "content": json.dumps(output) if isinstance(output, (dict, list)) else str(output),
                    })
                except Exception as exc:
                    logger.error(f"(🔧) Sync tool call {function_name_completed} failed: {exc}")
                    raise exc

        return messages

    async def _complete_tool_calling_cycle(self, response: ChatCompletion, messages: List[dict[str, str]]) -> ChatCompletion:
        """Handles the tool calling cycle."""
        assistant_message_dict = response.choices[0].message.model_dump()

        # Filter out unnecessary fields
        for field in ['reasoning', 'reasoning_details', 'refusal', 'annotations', 'audio']:
            if field in assistant_message_dict:
                del assistant_message_dict[field]

        messages.append(assistant_message_dict)
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
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": f"Error: Tool '{function_name}' not found."
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

    def _log_response(self, response: ChatCompletion) -> None:
        """Logs the full response, text response, reasoning, and updates token usage."""
        logger.debug(f"(📦) Full response: {response}")
        logger.debug(f"(✏️) Text response: {response.choices[0].message.content}")
        token_usage = getattr(response, 'usage', None)
        self._update_cumulative_token_usage(token_usage)
        reasoning = getattr(response.choices[0].message, 'reasoning', None)
        if reasoning:
            logger.debug(f"(🧠) Reasoning response: {reasoning}")
        else:
            logger.debug("(🧠) No reasoning provided in the message.")

    async def _generate_completition(self, messages: List[Dict], tools: Optional[Any] = None) -> ChatCompletion:
        """Generates a model completion using the provided messages and tools."""
        logger.debug(f"Adding the following settings: {self.settings}")
        logger.debug(f"Message is: {messages}")
        response = await self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    tools = tools if tools else None,
                    **self.settings
                )
        return response

    async def get_model_response(self,
                message: str,
                files_path: Optional[List[str]] = None) -> LLMResponse:
        """Sends a prompt to the LLM and returns the final response."""
        starting_time = time.time()
        self.number_of_interactions = 0
        logger.info(f"Starting prompt. {'Files included' if files_path else 'No files included.'} with model {self.model_name}")
        messages = []
        user_content = [{"type": "text", "text": message}]
        if files_path:
            processed_files = await self._process_files(files_paths=files_path)
            user_content.extend(processed_files)

        # Appending context
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
        processed_response =  self._process_response(response.choices[0].message.content)
        return processed_response