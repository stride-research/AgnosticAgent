
import logging

from pydantic import BaseModel, Field

from agnostic_agent import ToolkitBase
from agnostic_agent.utils import tool

logger = logging.getLogger(__name__)

class ChefToolkit(ToolkitBase):
    class GetIngredientsForDishSchema(BaseModel):
        dish: str = Field(..., description="The name of the dish. E.g., pizza")

    @tool(schema=GetIngredientsForDishSchema)
    def get_ingredients_for_dish(dish: str) -> dict:
        """Returns the ingredients for a given dish"""
        print(f"Executing get_ingredients_for_dish for {dish}")
        dummy_ingredients = {
            "pizza": ["dough", "tomato sauce", "cheese"],
            "salad": ["lettuce", "tomato", "cucumber"],
            "burger": ["bun", "beef patty", "lettuce", "cheese"]
        }
        return {"ingredients": dummy_ingredients.get(dish.lower(), ["unknown ingredient"])}

    class GetPriceForIngredientSchema(BaseModel):
        ingredient: str = Field(..., description="The name of the ingredient. E.g., cheese")

    @tool(schema=GetPriceForIngredientSchema)
    def get_ingredient_price(ingredient: str) -> dict:
        """Returns the price for a given ingredient"""
        print(f"Executing get_price_for_ingredient for {ingredient}")
        dummy_prices = {
            "dough": 2.0,
            "tomato sauce": 1.5,
            "cheese": 2.5,
            "lettuce": 1.0,
            "tomato": 1.2,
            "cucumber": 1.1,
            "bun": 1.3,
            "beef patty": 3.0
        }
        return {"price": dummy_prices.get(ingredient.lower(), 0.0)}

