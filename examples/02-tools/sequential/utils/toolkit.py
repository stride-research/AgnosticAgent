from pydantic import Field, BaseModel
from agentic_ai.utils import tool



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
