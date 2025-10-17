from fastapi import APIRouter, HTTPException, Depends, Request, Response, Security,UploadFile,File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.user import RegisterRequest, LoginRequest,ChangePasswordRequest,ChangePasswordInput, SuccessResponse, ErrorResponse
from fastapi.responses import FileResponse
from services.avatar_service import upload_avatar,get_avatar, update_avatar, delete_avatar
from datetime import datetime, timezone
from response_formatter import success_response, error_response
import os
from services.auth_service import(
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    revoke_token,
    change_password,
    is_valid_password
) 
from database.connection import db


router = APIRouter(prefix="/auth", tags=["Auth"])
collection = db["users"]
bearer_scheme = HTTPBearer(auto_error=False)

# ========== Helper: lấy next int _id ==========
async def get_next_user_id():
    last = await collection.find_one(sort=[("_id", -1)])
    return (last["_id"] + 1) if last else 1

# ========== REGISTER ==========
@router.post("/register")
async def register(req: RegisterRequest):
    if not is_valid_password(req.password):
        return error_response(
            "Password must be at least 6 characters, include uppercase, lowercase, number, and special character",
            status_code=400,
            code="INVALID_PASSWORD"
        )

    if await collection.find_one({"username": req.username}) or await collection.find_one({"email": req.email}):
        return error_response(
            "Username or email already exists",
            status_code=400,
            code="USER_EXISTS"
        )

    password_hash = hash_password(req.password)
    user_doc = {
        "_id": await get_next_user_id(),
        "username": req.username,
        "email": req.email,
        "password_hash": password_hash,
        "avatar": "",
        "created_at": datetime.now(timezone.utc),
        "subscriptions": [],
        "liked_videos": [],
        "watched_videos": [],
        "uploaded_videos": [],
        "notifications": [],
        "posts": []
    }
    await collection.insert_one(user_doc)

    return success_response("User registered successfully")

# ========== LOGIN ==========
@router.post("/login")
async def login(req: LoginRequest, response: Response):
    # Bắt buộc phải có email
    if not req.email:
        return error_response("Email is required", status_code=400, code="EMAIL_REQUIRED")

    # Tìm user theo email
    user = await collection.find_one({"email": req.email})

    # Không tìm thấy user hoặc sai mật khẩu
    if not user or not verify_password(req.password, user["password_hash"]):
        return error_response("Invalid credentials", status_code=401, code="INVALID_CREDENTIALS")

    # Tạo access token
    token = create_access_token(user_id=user["_id"], username=user["username"])

    # Lưu token vào cookie (HTTPOnly)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax"
        # secure=True  # bật nếu bạn chạy HTTPS
    )
    return success_response("Login successful", data={"access_token": token})

# ========== GET PROFILE ==========
@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"user": current_user}

#==================LOGOUT=============
@router.post("/logout")
def logout(request: Request, response: Response, credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    else:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    revoke_token(token)
    # xóa cookie phía client
    response.delete_cookie("access_token")
    return success_response("Logged out successfully")
#===================CHANGPASSWORD====================
# @router.post("/change-password",  response_model=SuccessResponse)
# async def change_user_password(
#     request: ChangePasswordInput,
#     current_user: dict = Depends(get_current_user)
# ):
#     req_with_user = ChangePasswordRequest(
#         username=current_user["username"],
#         old_password=request.old_password,
#         new_password=request.new_password
#     )

#     result = await change_password(req_with_user)
#     if result == -404:
#         raise HTTPException(status_code=404, detail="User not found")
#     if result == -1:
#         raise HTTPException(status_code=400, detail="Old password is incorrect")
#     if result == -2:
#         raise HTTPException(status_code=400, detail="Password must be at least 6 characters, include uppercase, lowercase, number, and special character")
#     if result == -3:
#         raise HTTPException(status_code=400, detail="New password must be different from old password")

#     # return {"message": "Password changed successfully"}
    


@router.post("/change-password")
async def change_user_password(
    request: ChangePasswordInput,
    current_user: dict = Depends(get_current_user)
):
    req_with_user = ChangePasswordRequest(
        username=current_user["username"],
        old_password=request.old_password,
        new_password=request.new_password
    )

    result = await change_password(req_with_user)
    if result == -404:
        raise HTTPException(status_code=404, detail="User not found")
    if result == -1:
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    if result == -2:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters, include uppercase, lowercase, number, and special character")
    if result == -3:
        raise HTTPException(status_code=400, detail="New password must be different from old password")

    return success_response("Password changed successfully")



#====================AVATAR==============================
# Upload avatar (cần token để xác thực)
# @router.post("/avatar/me")
# async def upload_my_avatar(
#     file: UploadFile = File(...),
#     current_user: dict = Depends(get_current_user)
# ):
#     return await upload_avatar(int(current_user["_id"]), file)

# Upload avatar (cần token để xác thực)
@router.post("/avatar/me")
async def upload_my_avatar(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # upload_avatar đã raise HTTPException khi lỗi → FE sẽ nhận JSON chuẩn
    result = await upload_avatar(int(current_user["_id"]), file)
    data = {"avatar_path": file.filename}
    return success_response("Avatar uploaded successfully", data=data)
   
# Hiển thị avatar (public)
@router.get("/avatar/{user_id}")
async def get_user_avatar(user_id: int):
    filepath = await get_avatar(user_id)
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Avatar not found")
    return FileResponse(filepath)


@router.put("/avatar/me")
async def update_my_avatar(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # update_avatar trả về đường dẫn hoặc tên file lưu trên server
    result = await update_avatar(int(current_user["_id"]), file)
    
    # trả về tên file gốc upload
    data = {"avatar_path": file.filename}
    return success_response("Avatar updated successfully", data=data)

#Delete avatar (cần token để xác thực)
@router.delete("/avatar/me")
async def delete_my_avatar(
    current_user: dict = Depends(get_current_user)
):
    result= await delete_avatar(int(current_user["_id"]))
    return success_response("Avatar delete successfully")
