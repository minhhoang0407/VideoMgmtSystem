from fastapi import APIRouter, HTTPException,UploadFile, File, Form,Depends
import json
from models.video import VideoInsertRequest,VideoResponse,VideoUpdateRequest,VideoInsertInput
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

@router.get("/")
async def list_videos():
    return await get_all_videos()

#cần token
#post metadata video
@router.post("/", response_model=VideoInsertRequest)
async def post_video(
    video_input: VideoInsertInput,
    current_user: dict = Depends(get_current_user)
):
    try:
        # ✅ Gán uploader từ token
        video_req = VideoInsertRequest(
            title=video_input.title,
            description=video_input.description,
            status=video_input.status,
            category=video_input.category,
            tags=video_input.tags
        )
        video_req.uploader_id = str(current_user["_id"])
        video_req.uploader_name = current_user["username"]

        created_video = await create_video(video_req)
        return created_video
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#post upload video
@router.post("/upload/{video_id}", response_model=VideoResponse)
async def upload_video_file(
    video_id: str,
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

@router.get("/{video_id}")
async def get_video(video_id: str):
    video = await get_video_by_id(video_id)
    if not video:
         # ❌ Không tìm thấy document -> 404
        raise HTTPException(status_code=404, detail="Video not found")
    return video

# 🔹 GET theo uploader_name
@router.get("/uploader/{uploader_name}", response_model=List[VideoResponse])
async def list_videos_by_uploader(uploader_name: str):
    videos =await get_videos_by_uploader(uploader_name)
    if videos==[]:
        raise HTTPException(status_code=404, detail="Video not found")
    else:
        return videos

# 🔹 GET theo category
@router.get("/category/{category}", response_model=List[VideoResponse])
async def list_videos_by_category(category: str):
    videos = await get_videos_by_category(category)
    if videos==[]:
        raise HTTPException(status_code=404, detail="Video not found")
    else:
        return videos

#cần token
# 🔹 PUT update video
@router.put("/{video_id}")
async def edit_video(video_id: str,
     video_update: VideoUpdateRequest,
     current_user: dict = Depends(get_current_user)
    ):
    return await update_video(video_id, video_update)

#cần token
@router.delete("/{video_id}")
async def remove_video(video_id: str,
     current_user: dict = Depends(get_current_user)
    ):
    return await delete_video(video_id)