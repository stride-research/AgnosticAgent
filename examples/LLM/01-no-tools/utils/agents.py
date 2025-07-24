import logging
 
from .schemas import Word

from academy.agent import Agent, action

from agentic_ai.LLM_agent import AIAgent, ResponseSettings

CORRECT_WORD = "CONSOLE"
TOPIC = "programming"
NUMBER_OF_ITERS = 2
    
logger = logging.getLogger(__name__)

class WordGuesser(Agent):
    attempted_words: list[str]
    thoughts: list[str]

    async def agent_on_startup(self) -> None:
        self.attempted_words = []
        self.thoughts = []
        self.LLMAgent = AIAgent(
            agent_name="WordGuesser",
            sys_instructions="You are a player of the famous wordle game. Explain what you do at each step",
            response_schema=Word,
            response_settings=ResponseSettings()
        )

    @action
    async def guess_word(self) -> str:

        message=f"Guess a {len(CORRECT_WORD)}-word letter. Topic of word is: {TOPIC}. Prior attempted words are: {self.attempted_words}. \
                        For prior attempts this has been the followed rationale/planned strategy: {self.thoughts}."
        
        llm_response = self.LLMAgent.prompt(message=message)
        logger.debug(f"Final response is: {llm_response.final_response}")
        logger.debug(f"Parsed response is: {llm_response.parsed_response}")
        #self.attempted_words.append(parsed_response)
        #self.thoughts.append(thoughts)
        return llm_response.parsed_response.guessed_word