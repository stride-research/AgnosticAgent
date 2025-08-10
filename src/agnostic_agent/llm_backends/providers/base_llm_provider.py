
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from agnostic_agent.utils import exception_controller_executor_instance

from ...utils.core.schemas import LLMResponse

logger = logging.getLogger(__name__)



class BaseLLMProvider(ABC):
      """
      FEATURES
            - Mandatory support
                  - LLM invokation
                  - Tool calling (multi-turn)
                  - Structured output
                  - Summary (usage, time to execute, nº interactions, etc) + LLM response logging (raw response, text response, reasoning)
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
      agent_name = ""
      model_name = ""
      sys_instructions = ""
      response_schema : Optional[Type[BaseModel]]= None
      tools = []
      
      async def prompt(self,
                   message: str, 
                   files_path: Optional[List[str]] = None
                   ) -> LLMResponse:
            output = await exception_controller_executor_instance.execute_with_retries(func=self.get_model_response,
                                                                                       message=message,
                                                                                       files_path=files_path)
            return output

      @abstractmethod
      async def get_model_response(self,
                   message: str, 
                   files_path: Optional[List[str]] = None) -> LLMResponse:
            pass

      @abstractmethod
      async def _generate_completition(self, messages: List[Dict], tools: Optional[Any] = None):
            pass

      @abstractmethod
      async def _complete_tool_calling_cycle(self, response, messages: List[dict[str, str]]):
            pass

      @abstractmethod
      def _log_response(self, response) -> None:
            pass

      @abstractmethod
      async def _process_files(self, files_paths: List[str]) -> List[Dict]:
            pass

      def _update_cumulative_token_usage(self, token_usage) -> None:
            """Updates the cumulative token usage statistics from the model response.

            Args:
                response (ChatCompletion): The model response containing usage information.
            """
            if token_usage:
                self.cumulative_token_usage['prompt_tokens'] += getattr(token_usage, 'prompt_tokens', 0)
                self.cumulative_token_usage['completion_tokens'] += getattr(token_usage, 'completion_tokens', 0)
                self.cumulative_token_usage['total_tokens'] += getattr(token_usage, 'total_tokens', 0)

      def _summary_log(self, starting_time: int) -> None:
            """Logs a summary of cumulative token usage, number of interactions, and elapsed time.

            Args:
                starting_time (int): The time when the prompt started.
            """
            logger.info(f"(💰) Cumulative token usage: prompt={self.cumulative_token_usage['prompt_tokens']}, completion={self.cumulative_token_usage['completion_tokens']}, total={self.cumulative_token_usage['total_tokens']}")
            logger.info(f"(🛠️) {self.number_of_interactions} interactions occured in function calling")
            if self.number_of_interactions == 0 and self.tools:
                logger.warning("The LLM hasnt invoked any function/tool, even tho u passed some tool definitions")
            logger.info(f"(⏱️) Took {round(time.time() - starting_time,2)} seconds to fullfill the given prompt")
      
      def _process_response(self, prompt_response: str) -> LLMResponse:
            """Processes the final ChatCompletion object to extract relevant data and log interactions.

            Args:
                response (ChatCompletion): The final model response.

            Returns:
                LLMResponse: The processed response containing the final and parsed responses.
            """
            logger.debug(f"Response is: {prompt_response}")

            
            if self.response_schema:
                json_dict = json.loads(prompt_response)
                parsed_data = self.response_schema.model_validate(json_dict)
            
            return LLMResponse(
                final_response=prompt_response,
                parsed_response=parsed_data if self.response_schema else None
            )


