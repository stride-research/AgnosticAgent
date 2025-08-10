from enum import Enum
from typing import Any, Callable, Optional, Type

import yaml
from pydantic import BaseModel

from agnostic_agent.config.config import CONFIG_DICT


class ToolSpec(BaseModel):
      func: Callable
      func_schema: Type[BaseModel]
      is_coroutine: bool

class LLMResponse(BaseModel):
    final_response: str
    parsed_response: Optional[Any] = None

extra_response_config = CONFIG_DICT["AI_agent"]["extra_response_settings"]

class ExtraResponseSettings(BaseModel):
    temperature: Optional[float] = extra_response_config["temperature"]
    max_tokens: Optional[int] = extra_response_config["max_tokens"]
    tool_choice: Optional[str] = extra_response_config["tool_choice"]

