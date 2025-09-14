from database.connection import db
from models.video import VideoResponse,VideoUpdateRequest,VideoInsertRequest
from services.upload_cloud_service import upload_file_to_cloud
from bson import ObjectId
from typing import List, Optional
from fastapi import HTTPException,UploadFile

#init collection
collection = db['videos']


def to_video(video: dict) -> VideoResponse:
    #Chuyển Mongo document thành Video model
    video["_id"] = str(video["_id"])  # ObjectId -> str
    return VideoResponse(**video)

#output service->list _id video
async def get_all_videos() -> List[VideoResponse]:
    videos:List[VideoResponse]=[]
    async for video in collection.find():
        videos.append(to_video(video))
    return videos
#------------------------------------------------------------------------------
#output service-> 
# 🔹 Service: tạo video chỉ lưu metadata
async def create_video(video_req):
    video_data = video_req.model_dump()
    result = collection.insert_one(video_data)
    video_data["_id"] = str(result.inserted_id)
    return video_data

# 🔹 Service: upload file và gắn vào video đã có (Cloudinary)
async def upload_video(video_id: str, file: UploadFile, uploader_name: str):
    video = collection.find_one({"_id": ObjectId(video_id)})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Kiểm tra quyền: chỉ chủ sở hữu mới được upload
    if video["uploader_name"] != uploader_name:
        raise HTTPException(status_code=403, detail="Not allowed")

    # Upload file lên Cloudinary
    try:
        file_url = await upload_file_to_cloud(file.file, resource_type="video")
    except Exception as e:
        #raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
        file_url="NaN"

    # Cập nhật DB
    collection.update_one(
        {"_id": ObjectId(video_id)},
        {"$set": {"file_url": file_url}}
    )

    video["file_url"] = file_url
    video["_id"] = str(video["_id"])
    return video
#------------------------------------------------------------------------------
#output service-> _id video
async def get_video_by_id(video_id: str) -> Optional[VideoResponse]:
    if not ObjectId.is_valid(video_id):
        return None
    video = await collection.find_one({"_id": ObjectId(video_id)})
    if video:
        return to_video(video)
    return None

#------------------------------------------------------------------------------
# 🔹 Lấy video theo uploader_name
async def get_videos_by_uploader(uploader_name: str, limit: int = 50) -> List[VideoResponse]:
    videos: List[VideoResponse] = []
    cursor = collection.find({"uploader_name": uploader_name}).sort("created_at", -1).limit(limit)
    async for video in cursor:
        videos.append(to_video(video))
    return videos

#------------------------------------------------------------------------------
# 🔹 Lấy video theo category
async def get_videos_by_category(category: str, limit: int = 50) -> List[VideoResponse]:
    videos: List[VideoResponse] = []
    cursor = collection.find({"category": category}).sort("created_at", -1).limit(limit)
    async for video in cursor:
        videos.append(to_video(video))
    return videos

#------------------------------------------------------------------------------
# PUT update video
async def update_video(video_id: str, video_update: VideoUpdateRequest):
    if not ObjectId.is_valid(video_id):
        raise HTTPException(status_code=400, detail="ID không hợp lệ")

    update_data = {k: v for k, v in video_update.model_dump().items() if v is not None}

    if not update_data:
        raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")

    result = await collection.update_one(
        {"_id": ObjectId(video_id)},
        {"$set": update_data}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Không thể cập nhật video")

    updated = await collection.find_one({"_id": ObjectId(video_id)})
    if not updated:
        raise HTTPException(status_code=500, detail="Video sau cập nhật không tìm thấy")

    return {"message":"updated"}

#------------------------------------------------------------------------------
# 🔹 DELETE video theo video_id
async def delete_video(video_id: str) -> bool:
    if not ObjectId.is_valid(video_id):
        raise HTTPException(status_code=400, detail="ID không hợp lệ")

    result = await collection.delete_one({"_id": ObjectId(video_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Video không tồn tại")

    return {"message":"updated"}