from pydantic import BaseModel, EmailStr,Field
from datetime import datetime
from typing import List, Optional
from typing import Optional, Any


class User(BaseModel):
    id: int = Field(..., alias="_id")                   # ✅ _id trong MongoDB bây giờ là số nguyên
    username: str
    email: EmailStr
    password_hash: str
    avatar: Optional[str] = None  # path cục bộ trong assets/avatars
    created_at: datetime = datetime.utcnow()
    subscriptions: List[int] = []   # ✅ nếu bạn muốn tham chiếu user khác → int thay vì str
    liked_videos: List[int] = []    # giả sử video cũng có _id int
    watched_videos: List[int] = []
    uploaded_videos: List[int] = []
    notifications: List[int] = []
    posts: List[int] = []
    

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    # username: Optional[str] = None
    #email: Optional[EmailStr] = None
    email: EmailStr
    password: str


class ChangePasswordInput(BaseModel):
    old_password: str
    new_password: str


class ChangePasswordRequest(ChangePasswordInput):
    username: Optional[str] = None  # sẽ được controller gán


class SuccessResponse(BaseModel):
    message: str
    data: Optional[Any] = None

class ErrorResponse(BaseModel):
    message: str
    code: str