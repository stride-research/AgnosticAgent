from typing import List, Type, Any
import logging

from .schemas import Interaction, InteractionType, StageType, FunctionAsTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)
      
class LLMProcessingLogs():
    """
      
      Stores in a list all interactions:
        - owner (user or model)
        - text / thought (only from model)
        - actual content
      
    """

    def __init__(self, agentName:str) -> None:
            self.agentName = agentName # The agent that instantiated the class
            self.LLMLogs = []
        
    def push_interaction(self, interaction: Interaction):
            self.LLMLogs.append(interaction)
    
    def pretty_print(self):
        """
        """
        print(f"\n🤖 LLM PROCESSING LOGS - {self.agentName} - All interactions recorded")
        print("=" * 60)
        
        if not self.LLMLogs:
            print("📭 No interactions logged yet")
            return
        
        for i, interaction in enumerate(self.LLMLogs, 1):
            # Stage indicator
            stage_emoji = "✅" if interaction.stage == StageType.final else "🔄"
            stage_text = "FINAL" if interaction.stage == StageType.final else "PROCESSING"
            
            # Owner indicator
            owner_emoji = "👤" if interaction.owner == "user" else "🤖"
            
            # Interaction type indicator
            type_emoji = {
                InteractionType.text: "💬",
                InteractionType.thought: "💭",
                InteractionType.function_call: "🔧",
                InteractionType.function_response: "📤"
            }[interaction.interaction_type]
            
            print(f"\n{i}. {stage_emoji} [{stage_text}] {owner_emoji} {interaction.owner.upper()}")
            print(f"TYPE: {type_emoji} {interaction.interaction_type.value.upper()}")
            
            # Format content based on type
            content = interaction.interaction_content
            if interaction.interaction_type == InteractionType.function_call:
                try:
                    import json
                    parsed = json.loads(content)
                    print(f"\t🔧 Function: {parsed.get('name', 'unknown')}")
                    print(f"\t📋 Args: {parsed.get('args', {})}")
                    print(f"\t")
                except:
                    print(f"\t📄 Content: {content}")
            elif interaction.interaction_type == InteractionType.thought:
                # Clean up thought content (remove markdown formatting)
                clean_content = content.replace("**", "").replace("\n\n", "\n")
                print(f"\t💭 {clean_content}")
            else:
                print(f"\t📄 {content}")
            print("- -" * 20)
        
        print(f"\n📊 Total Interactions: {len(self.LLMLogs)}")
        print("=" * 60)


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
    def __init__(self, funcs: dict[str, Type[FunctionAsTool]]) -> None:
        logger.debug(f"Provided functions for toolkit is: {funcs.keys()}")
        self.funcs = funcs
        self.tools = self.__set_up_tools()
    
    def __set_up_tools(self):
        tools = {}
        for func_name, tool in self.funcs.items():
            tools[func_name] = FunctionToTool(func=tool.func, func_schema=tool.func_schema)
        return tools

    def schematize(self):
        tool_schemas = []
        for name_tool, function_tool in self.tools.items():
             tool_schemas.append(function_tool.schematize())
        return tool_schemas

    def execute(self, func_name: str, **kwargs):
         return self.tools[func_name].execute(**kwargs)
    
tool_registry = {}
def tool(schema):
    def decorator(func: callable):
        tool_registry[func.__name__] = FunctionAsTool(func=func, func_schema=schema)
        return func
    return decorator