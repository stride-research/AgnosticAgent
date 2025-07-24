

from typing import List
from pydantic import BaseModel


class ORMResponseSchema(BaseModel):
      extracted_text: str

class Chunk(BaseModel):
      title: str
      description: str

class ChunkerResponseSchema(BaseModel):
      chunks: List[Chunk]