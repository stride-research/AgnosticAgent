import asyncio
import argparse
from .utils.schemas import Word
from agnostic_agent import LLMAgent
from ..config import inline_args

import logging

logger = logging.getLogger(__name__)

CORRECT_WORD = "CONSOLE"
TOPIC = "programming"
NUMBER_OF_ITERS = 2
AGENT_NAME = "WordGuesser"


async def run_example(backend:str, model:str):
    agent = LLMAgent(
        llm_backend=backend,
        agent_name=AGENT_NAME,
        model_name=model,
        sys_instructions="You are a player of the famous wordle game. Explain what you do at each step",
        response_schema=Word,
        tools=[]
    )
    message=f"Guess a {len(CORRECT_WORD)}-word letter. Topic of word is: {TOPIC}."

    for i in range(NUMBER_OF_ITERS):
        response = await agent.prompt(message=message)
        selected_word = response.parsed_response.guessed_word
        print(f"Selected word is: {selected_word}")
        if selected_word.strip().upper() == CORRECT_WORD:
            print("Word choice was correct")
            break
        else:
            print("Word choice was incorrect")

if __name__ == "__main__":
    asyncio.run(run_example(backend=inline_args.backend, 
                            model=inline_args.model))
