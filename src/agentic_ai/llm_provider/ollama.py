from .openai_provider import OpenAIProvider
import os

class Ollama(OpenAIProvider):
    def __init__(self,
                agent_name: str,
                model_name: str = "qwen3:8b",
                sys_instructions: str = None,
                response_schema: None = None,
                tools: list[str] = [],
                extra_response_settings: None = None,
                ) -> None:
        """Initializes the agent for use with Ollama.

        Args:
            agent_name (str): A name for the agent for logging purposes.
            model_name (str, optional): The LLM model to use. Defaults to "qwen3:8b".
            sys_instructions (str, optional): The system prompt for the model. Defaults to None.
            response_schema (Type[BaseModel], optional): A Pydantic model to structure the LLM's JSON output. Defaults to None.
            tools (List[str], optional): A list of tool names to use. Defaults to [].
            extra_response_settings (Type[ExtraResponseSettings], optional): Additional parameters for the OpenAI API call (e.g., temperature, max_tokens). Defaults to ExtraResponseSettings().
        """
        super().__init__(
            agent_name=agent_name,
            model_name=model_name,
            api_key="ollama",  # Ollama doesn't require a real API key
            base_url="http://localhost:11434/v1",
            sys_instructions=sys_instructions,
            response_schema=response_schema,
            tools=tools,
            extra_response_settings=extra_response_settings
        )