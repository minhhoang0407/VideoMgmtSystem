from pydantic import BaseModel, EmailStr
from datetime import datetime,timezone
from typing import List, Optional


class User(BaseModel):
    username: str
    email: EmailStr
    password_hash: str
    avatar: Optional[str] = None  # path cục bộ trong assets/avatars
    created_at: datetime = datetime.utcnow()
    subscriptions: List[str] = []
    liked_videos: List[str] = []
    watched_videos: List[str] = []
    uploaded_videos: List[str] = []
    notifications: List[str] = []
    posts: List[str] = []
    
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: str

class ChangePasswordInput(BaseModel):
    old_password: str
    new_password: str

class ChangePasswordRequest(ChangePasswordInput):
    username: str | None = None  # sẽ được controller gán