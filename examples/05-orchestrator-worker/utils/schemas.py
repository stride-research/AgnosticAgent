

from typing import List

from pydantic import BaseModel


class ORMResponseSchema(BaseModel):
      extracted_text: str

class ChunkNamed(BaseModel):
      title: str
      summary: str
      keyword: str

class ChunkNotNamed(BaseModel):
      summary: str
      keyword: str

class ChunkerResponseSchema(BaseModel):
      chunks: List[ChunkNamed]
