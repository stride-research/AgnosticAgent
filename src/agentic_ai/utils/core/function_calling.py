from typing import List, Type, Any, Dict
import logging

from .schemas import ToolSpec
from pydantic import BaseModel
import inspect

logger = logging.getLogger(__name__)


class RegisteredTool():
    """ Interface for scheamtize and executing a tool"""
    def __init__(self, function_as_tool: ToolSpec) -> None:
        self.func = function_as_tool.func
        self.func_schema = function_as_tool.func_schema
        self.name = function_as_tool.func.__name__
        self.description = function_as_tool.func.__doc__.strip() if function_as_tool.func.__doc__ else "No description provided."
        self.is_coroutine = function_as_tool.is_coroutine
        self.parameters_schema = self.func_schema.model_json_schema()
     
    def schematize(self) -> None:
         """
         Automatically fills in the expected schema for OpenAI tool calling
         """
         return {
              "type": "function",
              "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema # contains args type, descriptions and defaults
              }
         }
    
    def _execute_sync(self, **kwargs) -> Any:
        try:
            parsed_args = self.func_schema.model_validate(kwargs)
            result = self.func(**parsed_args.model_dump())
            return result
        except Exception as e:
            logger.error(f"Error executing SYNC tool '{self.name}' with args {kwargs}: {e}", exc_info=True)
            raise 

    async def _execute_async(self, **kwargs) -> Any:
        try:
            parsed_args = self.func_schema.model_validate(kwargs)
            result = await self.func(**parsed_args.model_dump())
            return result
        except Exception as e:
            logger.error(f"Error executing ASYNC tool '{self.name}' with args {kwargs}: {e}", exc_info=True)
            raise 
            
    def get_executable(self):
        """
        Returns the appropriate executable method based on whether the wrapped function is a coroutine.
        This is what you'll pass to executor.submit or asyncio.create_task.
        """
        if self.is_coroutine:
            return self._execute_async
        else:
            return self._execute_sync

class FunctionalToolkit:
    """
    Manages a collection of tools, providing methods to schematize and execute them.
    """
    def __init__(self, funcs: Dict[str, Type[ToolSpec]]) -> None:
        self.funcs = funcs
        self.tools: Dict[str, RegisteredTool] = self.__set_up_tools()
        self._tools_schemas_cache: List[Dict[str, Any]] = []

        if not self.funcs:
            logger.info("No functions for the toolkit have been provided (tool calling)")
        else:
            logger.info(f"Provided functions for the toolkit (tool calling) are: {list(self.funcs.keys())}")

    def __set_up_tools(self) -> Dict[str, RegisteredTool]:
        """
        Initializes RegisteredTool objects from raw function data.
        """
        tools = {}
        for func_name, function_as_tool in self.funcs.items():
            tools[func_name] = RegisteredTool(function_as_tool)
        return tools

    def schematize(self) -> List[Dict[str, Any]]:
        """
        Generates and caches the OpenAI tool schemas for all registered tools.
        """
        if not self._tools_schemas_cache:
            for function_tool in self.tools.values():
                self._tools_schemas_cache.append(function_tool.schematize())
        return self._tools_schemas_cache
    
tool_registry: Dict[str, ToolSpec] = {}
def tool(schema):
    def decorator(func: callable):
        is_coroutine = inspect.iscoroutinefunction(func)
        tool_registry[func.__name__] = ToolSpec(func=func, func_schema=schema, is_coroutine=is_coroutine)
        return func
    return decorator