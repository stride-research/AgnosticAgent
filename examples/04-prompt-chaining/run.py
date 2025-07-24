from .utils.schemas import LanguageSchema
from agentic_ai import AIAgent

import logging

logger = logging.getLogger(__name__)

file_path = []
PREFERRED_LANGUAGE = "English"

# 1) Make summarry of some text, 2) If not in english, translate to spanish 

def run_example(text: str):

      SumamarizerAgent = AIAgent(
                        agent_name="TextSummarizer",
                        sys_instructions="Summarize any text you receive",
                        tools=[]
      )

      summarizer_response = SumamarizerAgent.prompt(message=f"Summarize this text in its original language: {text}")

      LanguageDetectorAgent = AIAgent( # This could have been done in parallel but we are going for sequential for simplicity
                        agent_name="LanguageDetectorAgent",
                        sys_instructions="Detect the language of the text you receive",
                        response_schema=LanguageSchema,
                        tools=[]
      )

      languageDetector_response = LanguageDetectorAgent.prompt(message=f"What language is this in?: {summarizer_response.final_response}")

      logger.info(f"Language of summary is {languageDetector_response.parsed_response.language}")
      if languageDetector_response.parsed_response.language.strip().upper() != PREFERRED_LANGUAGE.strip().upper():
            logger.info(f"Text was NOT IN {PREFERRED_LANGUAGE}")
            TranslatorAgent = AIAgent(
                              agent_name="TextTranslator",
                              sys_instructions=f"Translate any text to {PREFERRED_LANGUAGE}",
                              tools=[]
            )

            summarizer_response = TranslatorAgent.prompt(message=f"Translatie this text: {text}")
            logger.debug(f"Summarized output is: {summarizer_response}")
      else:
            logger.info(f"Text was IN {PREFERRED_LANGUAGE}")
      logger.debug(f" Outline is: {summarizer_response}")

if __name__ == "__main__":
      text_spanish = """
            El viejo faro de Maspalomas parpadeaba, como un ojo cansado vigilando las dunas. Abajo, en la orilla, una niña llamada Sofía encontró una caracola que no era como las demás. Era lisa y de un azul tan profundo como el mar al atardecer.

            Cuando se la acercó al oído, no escuchó el sonido de las olas, sino una melodía suave, una canción que hablaba de barcos perdidos y estrellas antiguas. Cada día, volvía a la playa para escuchar su secreto, y la caracola siempre le cantaba una nueva estrofa. Nunca le contó a nadie sobre su tesoro, pues sabía que algunas magias solo existen cuando se guardan en el corazón."""
      text_english = """
                        The sun dipped below the horizon, painting the dunes of Maspalomas in shades of deep orange and violet. An old man named Mateo sat on the warm sand, his back against the gentle slope of a dune he’d known since childhood. He wasn't watching the tourists taking their final photos or the waves kissing the shore.

                        Instead, he watched a single, determined beetle climbing its own Everest of sand. Each time a gust of wind sent it tumbling down, the tiny creature righted itself and began its ascent again. Mateo smiled. For sixty years, he had watched the sun set from this very spot, seeing empires of sand rise and fall with the wind. The world changed, people came and went, but the small, stubborn beetle, like the dunes themselves, always endured. It was a quiet lesson in resilience, offered freely by the golden landscape each evening.
                     """
      
      run_example(text=text_spanish)
      run_example(text=text_english)

