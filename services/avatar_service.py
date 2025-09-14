import os
from fastapi import UploadFile
from bson import ObjectId
from database.connection import db 

AVATAR_DIR = "assets/avatars"
collection = db["users"]

# Đảm bảo thư mục tồn tại
os.makedirs(AVATAR_DIR, exist_ok=True)


async def upload_avatar(user_id: str, file: UploadFile):
    """
    Upload avatar cho user_id
    - Lưu file vào assets/avatars/{user_id}.png
    - Update path vào Mongo
    """
    # Tạo tên file (dùng user_id để tránh trùng)
    filename = f"{user_id}.png"
    filepath = os.path.join(AVATAR_DIR, filename)

    # Ghi file ra đĩa
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Update avatar path trong Mongo
    await collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"avatar": filepath}}
    )

    return {"message": "Avatar uploaded successfully", "avatar_path": filepath}


async def get_avatar(user_id: str):
    """
    Trả về đường dẫn file avatar dựa trên user_id
    """
    user = await collection.find_one({"_id": ObjectId(user_id)})
    if not user or not user.get("avatar"):
        return None
    return user["avatar"]

async def update_avatar(user_id: str, file: UploadFile):
    # Tìm avatar cũ
    user = await collection.find_one({"_id": ObjectId(user_id)})
    old_avatar = user.get("avatar") if user else None

    # Xóa file cũ nếu có
    if old_avatar and os.path.exists(old_avatar):
        os.remove(old_avatar)

    # Lưu file mới
    filename = f"{user_id}.png"
    filepath = os.path.join(AVATAR_DIR, filename)

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Cập nhật DB
    await collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"avatar": filepath}}
    )

    return {"message": "Avatar updated successfully"}

async def delete_avatar(user_id: str):
    user = await collection.find_one({"_id": ObjectId(user_id)})
    if not user or not user.get("avatar"):
        return {"message": "No avatar to delete"}

    filepath = user["avatar"]
    if os.path.exists(filepath):
        os.remove(filepath)

    await collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$unset": {"avatar": ""}}
    )
    return {"message": "Avatar deleted successfully"}