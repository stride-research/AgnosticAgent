# agentic_ai_framework

A modular framework for building agentic AI systems with advanced LLM orchestration, tool/function calling, and OpenRouter integration. Designed for researchers and developers to rapidly prototype and deploy AI agents with structured outputs, async support, and extensible toolkits.

## Features

- **OpenRouter & Anthropic patterns**: Out-of-the-box support for OpenRouter and Anthropic-style agent design.
- **Tool/function calling**: Register Python functions as tools for LLMs to call (OpenAI-compatible schema).
- **Structured outputs**: Use Pydantic schemas to enforce structured, type-safe LLM responses.
- **Async support**: Fully asynchronous agent execution for scalable workflows.
- **File support**: Agents can process and extract data from files.
- **Advanced logging**: Colorful, context-aware logging (with planned lineage and usage summaries).
- **CI pipeline**: Continuous integration for reliability.
- **Extensible toolkit**: Easily add your own tools and response schemas.
- **Linter included**: Code quality enforced.
- **Examples**: Prompt chaining, file upload, orchestrator-worker, and more.

## Installation

```bash
pip3 install -e .
```
- The `-e` flag is recommended for development.
- Requires Python 3.7+.
- Dependencies: `python-json-logger`, `openai`, `pydantic`, `dotenv` (see `requirements.txt`).

## Setup

1. **API Key**: Set your OpenRouter API key as an environment variable:
   ```bash
   export OPEN_ROUTER_API_KEY=your-api-key-here
   ```
2. **(Optional) Toolkits**: If using tool calling, ensure your toolkit modules are imported so functions are registered.

## Getting Started

Here's a minimal example of creating and running an agent:

```python
from agentic_ai import AIAgent
from pydantic import BaseModel

class Word(BaseModel):
    guessed_word: str

async def run_example():
    LLMAgent = AIAgent(
        agent_name="WordGuesser",
        sys_instructions="You are a player of the famous wordle game. Explain what you do at each step",
        response_schema=Word,
        tools=[]
    )
    message = "Guess a 7-letter word. Topic of word is: programming."
    response = await LLMAgent.prompt(message=message)
    print(response.parsed_response.guessed_word)
```

- See [`examples/`](examples/) for more advanced use cases (tool calling, prompt chaining, file upload, orchestrator-worker, etc).
- To run an example:
  ```bash
  python3 -m examples.01-no-tools.run
  ```

## Tool Calling Cycle

The framework's tool calling logic is based on the following flow:

<img width="776" height="541" alt="image" src="https://github.com/user-attachments/assets/7d9e9787-c43f-4c90-8f7f-857d2fa30dda" />

## Roadmap

- [x] OpenRouter support
- [x] Anthropic design patterns examples
- [x] File support
- [x] Linter
- [x] CI pipeline
- [x] Advanced LLM params
- [x] Async support
- [ ] Docs
- [ ] Logger (usage summary, agent lineage)
- [ ] Memory mechanisms
- [ ] Fault tolerance (retry on certain exceptions)
- [ ] Prompt caching
- [ ] Testing framework
- [ ] Ollama interface
- [ ] Benchmarking
- [ ] IMPRESS use case integration
- [ ] Support for DeepAgent


## License

MIT License. See [LICENSE](LICENSE) for details.

## Conceptualized Ideas
- Fine-tuning models (e.g., Qwen) for tool calling

