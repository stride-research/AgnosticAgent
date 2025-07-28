from agentic_ai import CONFIG_DICT

from enum import Enum
from typing import Any, Type, Optional, Callable

from pydantic import BaseModel
import yaml

class ToolSpec(BaseModel):
      func: Callable
      func_schema: Type[BaseModel]
      is_coroutine: bool

class LLMResponse(BaseModel):
    final_response: str
    parsed_response: Optional[Any] = None

extra_response_config = CONFIG_DICT["AI_AGENT"]["extra_response_settings"]

class ExtraResponseSettings(BaseModel):
    temperature: Optional[float] = extra_response_config["temperature"]
    max_tokens: Optional[int] = extra_response_config["max_tokens"]
    tool_choice: Optional[str] = extra_response_config["tool_choice"]

