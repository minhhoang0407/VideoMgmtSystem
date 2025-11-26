from fastapi import APIRouter, HTTPException,UploadFile, File, Form,Depends
import json
from models.video import VideoInsertRequest,VideoResponse,VideoUpdateRequest,VideoInsertInput
from response_formatter import success_response, error_response
from services.video_service import (
    get_all_videos,
    create_video,
    upload_video,
    get_video_by_id,
    get_videos_by_uploader,
    get_videos_by_category,
    update_video,
    delete_video)
from typing import List
from services.auth_service import get_current_user


router = APIRouter(prefix="/videos", tags=["Videos"])

# Hiển thị  all list các video
# @router.get("/")
# async def list_videos():
#     return await get_all_videos()

# @router.get("/")
# async def list_videos():
#     videos = await get_all_videos()
#     return {
#         "success": True,
#         "message": "Videos fetched successfully",
#         "data": videos
#     }
# Hiển thị list các video có phân trang
@router.get("/")
async def list_videos(limit: int = 6, skip: int = 0):
    videos = await get_all_videos(limit=limit, skip=skip)
    return success_response("Video fetched successfully", data=videos)


#cần token
#post metadata video
@router.post("/", response_model=VideoResponse)
async def post_video(
    video_input: VideoInsertInput,
    current_user: dict = Depends(get_current_user)
):
    try:
        # ✅ Tạo request để insert
        video_req = VideoInsertRequest(
            title=video_input.title,
            description=video_input.description,
            status=video_input.status,
            category=video_input.category,
            tags=video_input.tags
        )
        video_req.uploader_id = int(current_user["_id"])
        video_req.uploader_name = current_user["username"]

        # ✅ Tạo video trong DB
        created_video = await create_video(video_req)

        # ✅ Trả về VideoResponse
        return created_video
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#post upload video
@router.post("/upload/{video_id}", response_model=VideoResponse)
async def upload_video_file(
    video_id: int,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        updated = await upload_video(video_id, file, current_user["username"])
        return updated
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# from fastapi import APIRouter, HTTPException

# router = APIRouter()

#Video detail
# @router.get("/{video_id}")
# async def get_video(video_id: int):
#     video = await get_video_by_id(video_id)

#     if not video:
#         # ❌ Không tìm thấy video -> 404
#         raise HTTPException(status_code=404, detail="Video not found")

#     return {
#         "success": True,
#         "message": "Video fetched successfully",
#         "extensions": {
#             "code": "SUCCESS",
#             "status": 200,
#             "data": video
#         }
#     }

@router.get("/{video_id}")
async def get_video(video_id: int):
    video = await get_video_by_id(video_id)

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return success_response("Video fetched successfully", data=video)

# 🔹 GET theo uploader_name
@router.get("/uploader/{uploader_id}", response_model=List[VideoResponse])
async def list_videos_by_uploader(uploader_id: int):
    videos =await get_videos_by_uploader(uploader_id)
    if videos==[]:
        raise HTTPException(status_code=404, detail="Video not found")
    else:
        return videos

# 🔹 GET theo category
@router.get("/category/{category_name}", response_model=List[VideoResponse])
async def list_videos_by_category(category: str):
    videos = await get_videos_by_category(category)
    if videos==[]:
        raise HTTPException(status_code=404, detail="Video not found")
    else:
        return videos

#cần token
# 🔹 PUT update video
@router.put("/{video_id}")
async def edit_video(video_id: int,
     video_update: VideoUpdateRequest,
     current_user: dict = Depends(get_current_user)
    ):
    return await update_video(video_id, video_update)

#cần token
@router.delete("/{video_id}")
async def remove_video(video_id: int,
     current_user: dict = Depends(get_current_user)
    ):
    return await delete_video(video_id)