from pydantic import BaseModel
from enum import Enum
from typing import Any, Type, Optional

class InteractionType(str, Enum):
      text = "text"
      thought = "thought"
      function_call = "function_call"
      function_response = "function_response"

class StageType(str, Enum):
      final = "final"
      processing = "processing"

class Interaction(BaseModel):
      stage: StageType
      owner: str
      interaction_type: InteractionType
      interaction_content: str

class FunctionAsTool(BaseModel):
      func: Any
      func_schema: Type[BaseModel]

class LLMResponse(BaseModel):
    final_response: str
    parsed_response: Optional[Any] = None

class ExtraResponseSettings(BaseModel):
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 100_000

