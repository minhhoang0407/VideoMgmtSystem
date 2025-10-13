from database.connection import db
from models.video import VideoResponse, VideoUpdateRequest, VideoInsertRequest
from services.upload_cloud_service import upload_file_to_cloud
from typing import List, Optional
from fastapi import HTTPException, UploadFile

#init collection
collection = db['videos']

#=========================helper============================
def to_video(video: dict) -> VideoResponse:
    # Đảm bảo dict có _id
    return VideoResponse.model_validate(video, from_attributes=False)

async def get_next_id_from_max(col):
    doc = await col.find_one(sort=[("_id", -1)])   # lấy document có _id lớn nhất
    if doc and "_id" in doc:
        return int(doc["_id"]) + 1

#========================service=============================
#output service->list _id video
async def get_all_videos() -> List[VideoResponse]:
    videos:List[VideoResponse]=[]
    async for video in collection.find():
        videos.append(to_video(video))
    return videos
#------------------------------------------------------------------------------
#output service-> 
# 🔹 Service: tạo video chỉ lưu metadata
async def create_video(video_req: VideoInsertRequest):
    video_data = video_req.model_dump()
    # ✅ Lấy id mới
    new_id = await get_next_id_from_max(collection)
    if new_id is None:
        new_id = 100  # fallback id đầu tiên
    new_id = await get_next_id_from_max(collection)
    if new_id is None:
        new_id = 100
        
    video_data["_id"] = new_id

    # Thực hiện insert và lưu kết quả vào biến 'result'
    result = await collection.insert_one(video_data)
    # Tìm lại bản ghi hoàn chỉnh từ database bằng _id đã được tạo
    created_video_doc = await collection.find_one({"_id": result.inserted_id})
    # Kiểm tra xem có tìm thấy bản ghi không
    if not created_video_doc:
        raise HTTPException(status_code=500, detail="Failed to retrieve created video from database.")
    # Chuyển đổi bản ghi đầy đủ sang VideoResponse và trả về
    return to_video(created_video_doc)

# 🔹 Service: upload file và gắn vào video đã có (Cloudinary)
async def upload_video(video_id: int, file: UploadFile, uploader_name: str):
    video = await collection.find_one({"_id": video_id})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Kiểm tra quyền: chỉ chủ sở hữu mới được upload
    if video["uploader_name"] != uploader_name:
        raise HTTPException(status_code=403, detail="Not allowed")

    # Upload file lên Cloudinary
    try:
        file_url = await upload_file_to_cloud(file.file,uploader_name, resource_type="video")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
        file_url="NaN"

    # Cập nhật DB
    await collection.update_one(
        {"_id": video_id},
        {"$set": {"url": file_url}}
    )

    video["url"] = file_url
    return to_video(video)
#------------------------------------------------------------------------------
#output service-> _id video
async def get_video_by_id(video_id: int) -> Optional[VideoResponse]:
    video = await collection.find_one({"_id": video_id})
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
async def update_video(video_id: int, video_update: VideoUpdateRequest):
    update_data = {k: v for k, v in video_update.model_dump().items() if v is not None}

    if not update_data:
        raise HTTPException(status_code=400, detail="Không có dữ liệu để cập nhật")

    result = await collection.update_one(
        {"_id": video_id},
        {"$set": update_data}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Không thể cập nhật video")

    updated = await collection.find_one({"_id": video_id})
    if not updated:
        raise HTTPException(status_code=500, detail="Video sau cập nhật không tìm thấy")

    return {"message":"updated"}

#------------------------------------------------------------------------------
# 🔹 DELETE video theo video_id
async def delete_video(video_id: int) -> bool:
    result = await collection.delete_one({"_id": video_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Video không tồn tại")

    return {"message":"updated"}