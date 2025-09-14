import hashlib,hmac
from fastapi import HTTPException, Depends, Request, Security
from bson import ObjectId
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta,timezone
from config import HASH_KEY, JWT_SECRET, JWT_ALGORITHM
from database.connection import db
import re

# Bộ nhớ tạm để lưu token bị revoke (chỉ sống trong session server)
revoked_tokens = set()
collection = db["users"]
bearer_scheme = HTTPBearer(auto_error=False)

#=========================PASSWORD HASH=======================================
def hash_password(password: str) -> str:
    return hmac.new(HASH_KEY.encode(), password.encode(), hashlib.sha256).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    return hmac.compare_digest(hash_password(password), password_hash)

#=======================PASSWORD IS VALID=====================================
def is_valid_password(password: str) -> bool:
    """
    Validate mật khẩu theo tiêu chí:
    - >= 6 ký tự
    - Có ít nhất 1 chữ hoa
    - Có ít nhất 1 chữ thường
    - Có ít nhất 1 chữ số
    - Có ít nhất 1 ký tự đặc biệt
    """
    if len(password) < 6:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[@$!%*?&]", password):  # bạn có thể mở rộng ký tự đặc biệt
        return False
    return True
#==========================JWT TOKEN==========================================
def create_access_token(subject: str, expires_minutes: int = 60) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {"sub": subject, "exp": expire}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    # pyjwt có thể trả bytes trên 1 số phiên bản
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def verify_access_token(token: str):
    if token in revoked_tokens:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def revoke_token(token: str):
    revoked_tokens.add(token)

#===============CHANGEPASSWORD============================
async def change_password(req):
    user = await collection.find_one({"username": req.username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # kiểm tra mật khẩu cũ
    if not verify_password(req.old_password, user["password_hash"]):
        return -1

    # update mật khẩu mới
    new_hash = hash_password(req.new_password)
    collection.update_one(
        {"username": req.username},
        {"$set": {"password_hash": new_hash}}
    )
    return 1

#===============user helper===============================

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)  # dùng Security để OpenAPI nhận diện
):
    # ưu tiên Bearer token, fallback cookie 'access_token'
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    else:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # payload['sub'] ở đây ta dùng làm username (subject)
    username = payload.get("sub")
    user = await collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Chuyển ObjectId sang string
    user_out = {k: v for k, v in user.items() if k != "password_hash"}
    if "_id" in user_out and isinstance(user_out["_id"], ObjectId):
        user_out["_id"] = str(user_out["_id"])

    return user_out
