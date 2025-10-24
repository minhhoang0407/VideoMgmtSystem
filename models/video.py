from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class AIFeatures(BaseModel):
    transcript: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = []


# Video Response
class VideoResponse(BaseModel):
    id: int = Field(..., alias="_id")   # ✅ _id trong MongoDB là int
    title: str
    description: str
    uploader_id: Optional[int] = None   # ✅ nếu User._id cũng là int
    uploader_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "public"
    status_video: Optional[str]="Haven't Video"
    category: str
    tags: List[str] = []
    views: int = 0
    likes_count: int = 0
    likes: List[int] = []              # ✅ list user id (int)
    views_list: List[int] = []         # ✅ list user id (int)
    comments: List[dict] = []          # có thể sau này đổi sang Comment model
    ai_features: AIFeatures = Field(default_factory=AIFeatures)
    url: Optional[str] = None


class VideoInsertInput(BaseModel):
    title: str
    description: str
    status: str = "public"
    category: str
    tags: List[str] = []


# Video Insert Request
class VideoInsertRequest(VideoInsertInput):
    # ✅ Lấy từ token (controller set, user không nhập)
    uploader_id: Optional[int] = None
    uploader_name: Optional[str] = None

    # ✅ Các trường mặc định
    status_video: Optional[str]=None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    likes_count: int = 0
    likes: List[int] = []
    views_list: List[int] = []
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
