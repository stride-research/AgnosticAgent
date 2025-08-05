from .openai_provider import OpenAIProvider
import os

class OpenRouter(OpenAIProvider):
    def __init__(self, 
                agent_name: str,
                model_name: str = "google/gemini-2.5-pro",
                sys_instructions: str = None, 
                response_schema: None = None,
                tools: list[str] = [],
                extra_response_settings: None = None,
                ) -> None:
        """Initializes the agent for use with OpenRouter.

        Args:
            agent_name (str): A name for the agent for logging purposes.
            model_name (str, optional): The LLM model to use. Defaults to "anthropic/claude-sonnet-4".
            sys_instructions (str, optional): The system prompt for the model. Defaults to None.
            response_schema (Type[BaseModel], optional): A Pydantic model to structure the LLM's JSON output. Defaults to None.
            tools (List[str], optional): A list of tool names to use. Defaults to [].
            extra_response_settings (Type[ExtraResponseSettings], optional): Additional parameters for the OpenAI API call (e.g., temperature, max_tokens). Defaults to ExtraResponseSettings().

        Raises:
            ValueError: If the OpenRouter API key is not found in the environment variables.
        """
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not api_key:
            raise ValueError("Couldn't find OpenRouter's API key in OPEN_ROUTER_API_KEY environment variable.")
        
        super().__init__(
            agent_name=agent_name,
            model_name=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            sys_instructions=sys_instructions,
            response_schema=response_schema,
            tools=tools,
            extra_response_settings=extra_response_settings
        )