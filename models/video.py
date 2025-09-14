from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime,timezone

class AIFeatures(BaseModel):
    transcript: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = []

#video Response
class VideoResponse(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    title: str
    description: str
    uploader_id: Optional[str] = None
    uploader_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "public"
    category: str
    tags: List[str] = []
    views: int = 0
    likes_count: int = 0
    likes: List[str] = []
    views_list: List[str] = []
    comments: List[dict] = []
    ai_features: AIFeatures = AIFeatures()
    # ✅ Thuộc tính mới sau khi đã lưu lên cloud
    url: Optional[str] = None

class VideoInsertInput(BaseModel):
    title: str
    description: str
    status: str = "public"
    category: str
    tags: List[str] = []

#videoInsertRequest
class VideoInsertRequest(VideoInsertInput):
    # ✅ Lấy từ token (controller set, user không nhập)
    uploader_id: Optional[str] = None
    uploader_name: Optional[str] = None

    # ✅ Các trường mặc định
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    likes_count: int = 0
    likes: List[str] = []
    views_list: List[str] = []
    comments: List[dict] = []
    ai_features: AIFeatures = Field(default_factory=AIFeatures)
    url: Optional[str] = None


# Request khi update (PUT) → chỉ cho phép update 1 số field
class VideoUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


    
