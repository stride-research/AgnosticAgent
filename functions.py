
"""

HOW TO REACH INFINITE CYCLE?:
- Call function
- Add to context 
    - What context?
    - How much context?
- Recall with output
- Interaction logger


CONCLUSIONS
- You add to messages a list of dicts where each dicts represents a turn in the conversation
- Choices are different answers to the same output. For most cases we assume n=1
- The LLM's memory of the ongoing task comes primarly from the sequence of message you provide in the 'messages'
parameter

TO BE DONE:
- Test out parser on function calling 
- Do I even need to pass FunctionToolkit

"""



import enum
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from pydantic import Field, BaseModel
from agentic_ai.LLM_agent import FunctionsToToolkit, AIAgent
from agentic_ai.LLM_agent import tool_registry, tool, FunctionsToToolkit
import logging

logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(
      api_key=os.getenv("OPEN_ROUTER_API_KEY"),
      base_url="https://openrouter.ai/api/v1"
)

class GetCurrentTemperatureSchema(BaseModel):
    location: str = Field(..., description="The city and state. E.g., SFO")
    unit: str = Field("fahrenheit", description="The unit to use")

@tool(schema=GetCurrentTemperatureSchema)
def get_current_temperature(location: str, unit: str = "fahrenheit") -> dict:
    """ Returns the given temperature for a given location"""
    print(f"Executing get_current_weather for {location} with unit {unit}")
    return {"temperature": 25, "unit": unit}

class GetCurrentHumiditySchema(BaseModel):
    location: str = Field(..., description="The city and state. E.g., SFO")

@tool(schema=GetCurrentHumiditySchema)
def get_current_humidity(location: str) -> dict:
    """ Returns the current humidity for a given location"""
    print(f"Executing get_current_humidity for {location}")
    return {"humidity": 60}


LLM = AIAgent(
                              agent_name="Weather guy",
                              model_name="google/gemini-2.5-pro",
                              sys_instructions="You are a helpful assistant. Use the tools provided in order to fullfill an user's request."
                          )

response = LLM.prompt("What is the weather (temperature and humidity) like in San Francisco?")

print(f"Final response: {response}")
