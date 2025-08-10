import inspect
import logging
from typing import Any, Dict, List, Type

from ..schemas import ToolSpec

logger = logging.getLogger(__name__)


class RegisteredTool():
    """Represented a registered tool inside the toolkit

    Attributes:
        func: the callable method
        func_schema: pydantic schema defining the arguments
        name: name of the callable method
        description: docstring of the method
        is_coroutine: defines if the given callable is async or not
        parameters_schema: extract the json schema out of the Pydantic model
    
    """
    def __init__(self, function_as_tool: ToolSpec) -> None:
        """Initializes the instances with the function being passed.

        Args:
            function_as_tool: contains function callable + metadata (is_coroutine & schema)
        """
        self.func = function_as_tool.func
        self.func_schema = function_as_tool.func_schema
        self.name = function_as_tool.func.__name__
        self.description = function_as_tool.func.__doc__.strip() if function_as_tool.func.__doc__ else "No description provided."
        self.is_coroutine = function_as_tool.is_coroutine
        self.parameters_schema = self.func_schema.model_json_schema()
     
    def schematize(self) -> Dict:
         """Automatically fills in the expected schema for OpenAI tool calling

         Args:
            None
         
         Returns:
            A dic mapping a function callable to its name, description and paramteres schema
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
        """Executes a given method in synchronous manner.

        Validates passed kwargs, calls the function with provided params and returns result.

        Returns:
            Output of the called method

        Raises:
            Exception: General error occured during execution
        """
        try:
            parsed_args = self.func_schema.model_validate(kwargs)
            result = self.func(**parsed_args.model_dump())
            return result
        except Exception as e:
            logger.error(f"Error executing SYNC tool '{self.name}' with args {kwargs}: {e}", exc_info=True)
            raise 

    async def _execute_async(self, **kwargs) -> Any:
        """Executes a given method in an asynchronous manner.

        Validates passed kwargs, calls the function with provided params and returns result.

        Raises:
            Exception: General error occured during execution
        """
        try:
            parsed_args = self.func_schema.model_validate(kwargs)
            result = await self.func(**parsed_args.model_dump())
            return result
        except Exception as e:
            logger.error(f"Error executing ASYNC tool '{self.name}' with args {kwargs}: {e}", exc_info=True)
            raise 
            
    def get_executable(self) -> None:
        """Returns the appropriate executable method based on whether the wrapped function is a coroutine.
        Servers as router for calling the proper execution procedures adjusting automatically for coroutine 
        to normal functions.
        """
        if self.is_coroutine:
            return self._execute_async
        else:
            return self._execute_sync



class FunctionalToolkit:
    """Manages a collection of tools, providing methods to schematize and execute them.
    
    Attributes:
        funcs: passed dictionary of callables with name as key and ToolSpec as value
        tools: registered tools dict (transformed funcs input)
        _tools_schema_cache: rstores schema of a given tool for the times its being called to schematize 
        more than once for the same instance

    """
    def __init__(self, funcs: Dict[str, Type[ToolSpec]]) -> None:
        """Initializes instances with funcs dictionary

        Args:
            funcs: passed dictionary of callables with name as key and ToolSpec as value
        """
        self.funcs = funcs
        self.tools: Dict[str, RegisteredTool] = self.__set_up_tools()
        self._tools_schemas_cache: List[Dict[str, Any]] = []

        if not self.funcs:
            logger.info("No functions for the toolkit have been provided (tool calling)")
        else:
            logger.info(f"Provided functions for the toolkit (tool calling) are: {list(self.funcs.keys())}")

    def __set_up_tools(self) -> Dict[str, RegisteredTool]:
        """Initializes RegisteredTool objects from raw function data.

        Returns:
            Dictionary with registered tools as value and function name as key
        """
        tools = {}
        for func_name, function_as_tool in self.funcs.items():
            tools[func_name] = RegisteredTool(function_as_tool)
        return tools

    def schematize(self) -> List[Dict[str, Any]]:
        """Generates and caches the OpenAI tool schemas for all registered tools.

        Returns:
            List of all the functions' schemas 
        """
        if not self._tools_schemas_cache:
            for function_tool in self.tools.values():
                self._tools_schemas_cache.append(function_tool.schematize())
        return self._tools_schemas_cache
    
tool_registry: Dict[str, ToolSpec] = {}
def tool(schema):
    """Decorator that write tos a variable for registering methods automatically.

    Args:
        schema: Pydantic model defining args with datatype, default and descriptions
    """
    def decorator(func: callable):
        is_coroutine = inspect.iscoroutinefunction(func)
        tool_registry[func.__name__] = ToolSpec(func=func, func_schema=schema, is_coroutine=is_coroutine)
        return func
    return decorator

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