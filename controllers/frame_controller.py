from fastapi import APIRouter, HTTPException, UploadFile, File, Query, status
from typing import List, Dict, Any

from services.search_service import search_service
from database.connection import db

FRAMES_COLLECTION = "frames"

router = APIRouter(
    prefix="/frames",
    tags=["Frames Search"]
)

# --- Hàm helper để tránh lặp code ---
def serialize_doc(doc):
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

async def _fetch_and_sort_frames(frame_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Hàm nhận danh sách frame_id, truy vấn DB và trả về danh sách document
    đã được sắp xếp đúng thứ tự.
    """
    if not frame_ids:
        return []
        
    frames_collection = db[FRAMES_COLLECTION]
    cursor =  frames_collection.find({"frame_id": {"$in": frame_ids}})
    
    # Tạo một map để giữ đúng thứ tự từ kết quả tìm kiếm
    results_map = {doc["frame_id"]: serialize_doc(doc) async for doc in cursor}
    
    # Sắp xếp kết quả theo đúng thứ tự mà FAISS đã trả về
    sorted_results = [results_map[fid] for fid in frame_ids if fid in results_map]
    return sorted_results

# --- API Endpoints ---

@router.get("/search", response_model=List[Dict[str, Any]])
async def search_frames_by_text(
    text: str = Query(..., min_length=3, description="Nội dung văn bản cần tìm kiếm")
):
    """
    API tìm kiếm keyframes dựa trên nội dung văn bản.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Nội dung tìm kiếm không được để trống.")
    
    try:
        found_frame_ids = await search_service.search_by_text(text, top_k=10)
        return await _fetch_and_sort_frames(found_frame_ids)

    except Exception as e:
        print(f"ERROR: Text search failed: {e}")
        raise HTTPException(status_code=500, detail="Đã xảy ra lỗi trong quá trình tìm kiếm bằng văn bản.")


@router.post("/image-search", response_model=List[Dict[str, Any]])
async def search_frames_by_image(
    image: UploadFile = File(..., description="File ảnh cần tìm kiếm")
):
    """
    API tìm kiếm keyframes dựa trên một hình ảnh mẫu.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên không phải là hình ảnh.")

    try:
        image_bytes = await image.read()
        found_frame_ids = await search_service.search_by_image(image_bytes, top_k=10)
        return await _fetch_and_sort_frames(found_frame_ids)

    except Exception as e:
        print(f"ERROR: Image search failed: {e}")
        raise HTTPException(status_code=500, detail="Đã xảy ra lỗi trong quá trình tìm kiếm bằng hình ảnh.")

#========================Hide API=========================
@router.post("/reload-index", status_code=status.HTTP_200_OK)
async def reload_search_index():
    """
    API để làm mới dữ liệu tìm kiếm (Hot Reload) sau khi có video mới.
    Nên được gọi bởi Worker hoặc Admin.
    """
    try:
        print("🔄 Yêu cầu Reload Index nhận được...")
        # Gọi hàm reload_indices() của service để đọc lại file .npy và .json từ đĩa lên RAM
        await search_service.reload_indices()
        
        # Lấy thông số hiện tại để trả về
        total_vectors = search_service.faiss_index.ntotal if search_service.faiss_index else 0
        return {
            "message": "Index reloaded successfully",
            "total_frames_in_ram": total_vectors
        }
    except Exception as e:
        print(f"❌ Reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload index: {str(e)}")