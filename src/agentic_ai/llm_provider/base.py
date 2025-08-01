
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict

from ..utils.core.schemas import LLMResponse

class LLMProvider(ABC):
      """
      FEATURES
            - Mandatory support
                  - LLM invokation
                  - Tool calling (multi-turn)
                  - Structured output
                  - Usage + response logging
            - Encouraged support 
                  - File upload
      
      """
      number_of_interactions = 0
      interactions_limit = 10 
      cumulative_token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
      }
      @abstractmethod
      async def prompt(self,
                   message: str, 
                   files_path: Optional[List[str]] = None,
                   n_of_attempts: Optional[int] = 2) -> LLMResponse:
            pass

      @abstractmethod
      async def _generate_completition(self, messages: List[Dict], tools: Optional[Any] = None):
            pass

      @abstractmethod
      async def _complete_tool_calling_cycle(self, response, messages: List[dict[str, str]]):
            pass

      @abstractmethod
      async def _process_files(self, files_paths: List[str]) -> List[Dict]:
            pass

      @abstractmethod
      def _summary_log(self, starting_time: int) -> None:
            pass

      @abstractmethod
      def _log_response(self, response) -> None:
            pass


