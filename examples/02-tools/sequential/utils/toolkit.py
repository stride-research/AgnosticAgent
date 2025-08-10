from pydantic import BaseModel, Field

from agnostic_agent import ToolkitBase
from agnostic_agent.utils import tool


class MathematicianToolkit(ToolkitBase):
    class AdditionSchema(BaseModel):
        x: int = Field(..., description="First operand")
        y: int = Field(..., description="Second operand")

    @tool(schema=AdditionSchema)
    def addition(x: int, y: int) -> dict:
        """
        Do addition
        """
        return {"result": x + y}

    class MultiplicationSchema(BaseModel):
        x: int = Field(..., description="First operand")
        y: int = Field(..., description="Second operand")

    @tool(schema=MultiplicationSchema)
    def multiplication(x: int, y: int) -> dict:
        """
        Do multiplication
        """
        return {"result": x * y}
    
    def extract_tools_names(self):
        return super().extract_tools_names()
        

    


