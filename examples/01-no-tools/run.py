from .utils.schemas import Word
from agentic_ai import AIAgent
from agentic_ai.utils import add_context_to_log

import logging



logger = logging.getLogger(__name__)

CORRECT_WORD = "CONSOLE"
TOPIC = "programming"
NUMBER_OF_ITERS = 2
AGENT_NAME = "WordGuesser"


def run_example():
    LLMAgent = AIAgent(
                agent_name=AGENT_NAME,
                sys_instructions="You are a player of the famous wordle game. Explain what you do at each step",
                response_schema=Word
            )
    message=f"Guess a {len(CORRECT_WORD)}-word letter. Topic of word is: {TOPIC}."

    for i in range(NUMBER_OF_ITERS):
        response = LLMAgent.prompt(message=message)
        selected_word = response.parsed_response.guessed_word
        print(f"Selected word is: {selected_word}")
        if selected_word.strip().upper() == CORRECT_WORD:
            print("Word choice was correct")
            break
        else:
            print("Word choice was incorrect")

if __name__ == "__main__":
    with add_context_to_log(agent_name=AGENT_NAME):
        run_example()
    