

from .import toolkit
from agentic_ai.LLM_agent import AIAgent
from academy.agent import Agent, action

from pydantic import BaseModel



class WeatherGuy(Agent):
      async def agent_on_startup(self):
            self.LLM = AIAgent(
                              agent_name="Weather guy",
                              model_name="google/gemini-2.5-pro",
                              sys_instructions="You are a helpful assistant. Use the tools provided in order to fullfill an user's request."
                          )
      
      @action
      async def get_weather(self):
           response = self.LLM.prompt("What is the weather (temperature and humidity) like in San Francisco?")
           return response


