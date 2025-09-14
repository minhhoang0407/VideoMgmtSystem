from pydantic import BaseModel, Field
from typing import Optional

class CategoryRequest(BaseModel):
    name: str = Field(...)
    description: Optional[str]

class CategoryResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
