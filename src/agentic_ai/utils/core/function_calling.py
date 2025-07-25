from typing import List, Type, Any
import logging

from .schemas import Interaction, InteractionType, StageType, FunctionAsTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FunctionToTool():
    def __init__(self, func: callable, func_schema: Type[BaseModel]) -> None:
          self.func = func 
          self.func_schema = func_schema
          self.name = self.func.__name__
          self.description = self.func.__doc__.strip()
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

    def execute(self, **kwargs) -> Any:
         parsed_args = self.func_schema.model_validate(kwargs)
         return self.func(**parsed_args.model_dump())

class FunctionsToToolkit():
    """ 
    Note: no two functions can be called the same
    """
    tools_schemas = []
    def __init__(self, funcs: dict[str, Type[FunctionAsTool]]) -> None:
        if funcs:
            logger.info(f"Provided functions for the toolkit (tool calling) is: {funcs.keys()}")
        else:
            logger.info(f"No functions for the toolkit have been provided (tool calling)")
        self.funcs = funcs
        self.tools = self.__set_up_tools()
    
    def __set_up_tools(self):
        tools = {}
        for func_name, tool in self.funcs.items():
            tools[func_name] = FunctionToTool(func=tool.func, func_schema=tool.func_schema)
        return tools

    def schematize(self):
        if not self.tools_schemas:
            for name_tool, function_tool in self.tools.items():
                self.tools_schemas.append(function_tool.schematize())
            return self.tools_schemas
        else:
            return self.tools_schemas
        
    def execute(self, func_name: str, **kwargs):
         return self.tools[func_name].execute(**kwargs)
    
tool_registry = {}
def tool(schema):
    def decorator(func: callable):
        tool_registry[func.__name__] = FunctionAsTool(func=func, func_schema=schema)
        return func
    return decorator