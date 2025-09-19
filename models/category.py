from pydantic import BaseModel, Field
from typing import Optional


class CategoryRequest(BaseModel):
    name: str = Field(...)
    description: Optional[str] = None


class CategoryResponse(BaseModel):
    id: int = Field(..., alias="_id")   # ✅ _id trong Mongo là int
    name: str
    description: Optional[str] = None
