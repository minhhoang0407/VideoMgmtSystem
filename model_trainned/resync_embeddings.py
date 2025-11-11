import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
import requests
from io import BytesIO
import asyncio

# Thêm sys.path để import database connection
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import db

# ============================
# Config
# ============================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
# TẠO RA MỘT FILE .npy MỚI ĐỂ TRÁNH GHI ĐÈ LÊN FILE CŨ NGAY LẬP TỨC
NEW_EMBEDDINGS_NPY = os.path.join(BASE_DIR, "cached_frame_embs_RESYNCED.npy")

FRAMES_COLLECTION = "frames"
BATCH_SIZE = 32 # Xử lý theo lô để tận dụng GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Main Logic
# ============================
async def resync_embeddings_from_db():
    print("--- Bắt đầu quá trình đồng bộ lại embeddings từ MongoDB và Cloud ---")
    
    # 1. Khởi tạo model CLIP
    print("Initializing CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", quick_gelu=True
    )
    clip_model = clip_model.to(DEVICE).eval()
    print("CLIP model loaded.")

    # 2. Lấy danh sách frame đã được sắp xếp từ MongoDB
    print(f"Fetching sorted frame documents from '{FRAMES_COLLECTION}' collection...")
    frames_collection = db[FRAMES_COLLECTION]
    
    # Sắp xếp theo video_id và frame_id để có một trật tự nhất quán
    cursor = frames_collection.find({}, {"path": 1}).sort([
        ("video_id", 1), 
        ("frame_id", 1)
    ])
    
    # Lấy toàn bộ document vào một danh sách
    sorted_frames = await cursor.to_list(length=None)
    
    if not sorted_frames:
        print("❌ Collection 'frames' rỗng. Không có gì để xử lý.")
        return
        
    print(f"✅ Tìm thấy {len(sorted_frames)} frames cần xử lý.")

    # 3. Tải ảnh, encode lại và lưu vào danh sách
    all_new_embeddings = []
    
    pbar = tqdm(total=len(sorted_frames), desc="🖼️  Re-encoding frames")

    for i in range(0, len(sorted_frames), BATCH_SIZE):
        batch_docs = sorted_frames[i : i + BATCH_SIZE]
        image_tensors = []
        
        for doc in batch_docs:
            image_url = doc.get("path")
            if not image_url:
                pbar.update(1)
                continue
                
            try:
                # Tải ảnh từ cloud
                response = requests.get(image_url, timeout=100)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                
                # Preprocess ảnh và thêm vào batch
                image_tensors.append(clip_preprocess(image))
            except Exception as e:
                print(f"\n⚠️ Lỗi khi xử lý ảnh {image_url}: {e}")
        
        if image_tensors:
            # Tạo tensor từ batch ảnh
            batch_tensor = torch.stack(image_tensors).to(DEVICE)
            
            # Encode cả batch trong một lần
            with torch.no_grad():
                feats = clip_model.encode_image(batch_tensor)
                feats /= feats.norm(dim=-1, keepdim=True)
                all_new_embeddings.append(feats.cpu().numpy())
        
        pbar.update(len(batch_docs))
        
    pbar.close()

    if not all_new_embeddings:
        print("❌ Không encode được embedding nào.")
        return

    # 4. Nối tất cả các batch embedding và lưu file .npy mới
    final_embeddings = np.concatenate(all_new_embeddings, axis=0)
    print(f"\n💾 Đang lưu file embedding đã đồng bộ tại: {NEW_EMBEDDINGS_NPY}")
    np.save(NEW_EMBEDDINGS_NPY, final_embeddings)
    
    print("\n🎉 HOÀN TẤT! Quá trình đồng bộ đã xong.")
    print(f"   - Tổng số embedding đã tạo: {len(final_embeddings)}")
    print("   - Bạn có thể đổi tên file 'cached_frame_embs_RESYNCED.npy' thành 'cached_frame_embs_synced.npy' để hệ thống sử dụng.")


if __name__ == "__main__":
    # Cần cài đặt requests: pip install requests
    asyncio.run(resync_embeddings_from_db())

# note: ảnh lỗi truy cập cần xử lí
#//res.cloudinary.com/dbdty9p1v/image/upload/v1760703569/frames/v0zbmq4dhhzqn464kbpl.jpg   #6432
