import logging
from typing import Any, List, Optional, Type

from pydantic import BaseModel

from agnostic_agent import BaseLLMProvider, OllamaClient, OpenRouterClient
from agnostic_agent.utils import add_context_to_log

from .utils.core.schemas import ExtraResponseSettings, LLMResponse

logger = logging.getLogger(__name__)

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
            
            self.agent_name = agent_name
            self.model_name = model_name
            
            self.llm_backend = self._resolve_llm_backend_object(
                  llm_backend=llm_backend,
                  agent_name=agent_name,
                  model_name=model_name,
                  sys_instructions=sys_instructions,
                  response_schema=response_schema,
                  tools=tools,
                  extra_response_settings=extra_response_settings
            )

      def _resolve_llm_backend_object(self, 
                                    llm_backend: str,
                                    **kwargs) -> BaseLLMProvider: # simple factory pattern
            logger.debug(f"KWARGS IS: {kwargs}")
            llm_provider = llm_backend.strip().lower()
            if llm_provider == "openrouter":
                  return OpenRouterClient(**kwargs)
            elif llm_provider == "ollama":
                  return OllamaClient(**kwargs)
            else: 
                  raise ValueError(f"Unknown LLM backend: {llm_backend}")

      
      async def prompt(self,
                    message: str,  
                    files_path: Optional[List[str]] = None) -> LLMResponse:  # strategy pattern
            with add_context_to_log(agent_name=self.agent_name, model_name=self.model_name, llm_backend=self.llm_backend):
                  result = await self.llm_backend.prompt(message=message,
                                                files_path=files_path)
                  logger.debug(f"Final text response is: {result.final_response}")
                  logger.debug(f"Final parsed response is: {result.parsed_response}")
            return result
      