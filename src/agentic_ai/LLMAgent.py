
"""
- Interface/Implementation: LLM provider (OpenRouter, Google, )
- Abstraction/Subject: 

We are using this pattern not because of the combinationatorial explosion, but rather because of the 
the unified interface and its integration via object composition

"""

from typing import Optional, List, Type, Any

from .utils.core.schemas import ExtraResponseSettings, LLMResponse
from .llm_provider.base import LLMProvider
from .llm_provider.open_router import OpenRouter
from .llm_provider.ollama import Ollama



from pydantic import BaseModel


class LLMAgent():
      def __init__(self, 
                    llm_backend: str,
                    agent_name: str,
                    model_name: str = "google/gemini-2.5-pro",
                    sys_instructions: Optional[str] = None, 
                    response_schema: Optional[Type[BaseModel]] = None,
                    tools: Optional[List[Any]] = [],
                    extra_response_settings: Optional[Type[ExtraResponseSettings]] = ExtraResponseSettings(),
                    ) -> None:
            self.llm_backend = self._resolve_llm_backend_object(llm_backend)
            self.agent_name = agent_name
            self.model_name = model_name
            self.sys_instructions = sys_instructions
            self.response_schema = response_schema
            self.tools = tools
            self.extra_response_settings = extra_response_settings

      def _resolve_llm_backend_object(self, llm_backend : str) -> LLMProvider:
            match llm_backend.strip().lower():
                  case "openrouter":
                        return OpenRouter()
                  case "ollama":
                        return Ollama()
      
      async def prompt(self,
                    message: str, 
                    files_path: Optional[List[str]] = None,
                    n_of_attempts: Optional[int] = 2) -> LLMResponse:
            future = self.llm_backend.prompt(message, files_path, n_of_attempts) 
            return await future