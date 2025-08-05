
from agnostic_agent.utils import tool
from agnostic_agent import ToolkitBase

import logging

from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)

class WeatherToolkit(ToolkitBase):
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

logger.info("All tools have been succesfully loaded")