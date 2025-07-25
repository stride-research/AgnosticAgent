from pydantic import BaseModel
from enum import Enum
from typing import Any, Type, Optional, Callable

class ToolSpec(BaseModel):
      func: Callable
      func_schema: Type[BaseModel]
      is_coroutine: bool

class LLMResponse(BaseModel):
    final_response: str
    parsed_response: Optional[Any] = None

class ExtraResponseSettings(BaseModel):
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = 30_000
    tool_choice: Optional[str] = "auto" # or 'required'

