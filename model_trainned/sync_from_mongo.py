import json
import numpy as np
import os
from tqdm import tqdm
import sys
import asyncio # << 1. IMPORT ASYNCIO

# Thêm dòng này để đi ngược lên 1 cấp (từ model_trainned ra backend_app)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import db

# ============================
# Config
# ============================
BASE_DIR = r"D:\KLCN\backend_app\model_trainned"
COLLECTION_NAME = "frames"
OLD_EMB_PATH = os.path.join(BASE_DIR, "cached_frame_embs.npy")
OLD_META_PATH = os.path.join(BASE_DIR, "cached_frame_meta.json")
NEW_EMB_OUTPUT_PATH = os.path.join(BASE_DIR, "cached_frame_embs_synced.npy")

# ============================
# Main Logic
# ============================
async def sync_embeddings_from_mongo():
    print("Bắt đầu quá trình đồng bộ embedding từ MongoDB...")

    # 1. Kết nối và lấy dữ liệu từ MongoDB
    try:
        collection = db[COLLECTION_NAME]

        print(f"-> Đang lấy dữ liệu từ collection '{COLLECTION_NAME}'...")
        mongo_docs = []
        cursor = collection.find({}, {"frame_id": 1})
        # This async for loop is correct for your async db connection
        async for document in cursor:
            mongo_docs.append(document)
        
        # client.close()  # << 2. XÓA DÒNG NÀY VÌ 'client' KHÔNG TỒN TẠI

        if not mongo_docs:
            print(f"❌ Collection '{COLLECTION_NAME}' rỗng hoặc không tồn tại.")
            return

        valid_frame_ids = {doc['frame_id'] for doc in mongo_docs}
        print(f"-> Tìm thấy {len(valid_frame_ids)} frame ID hợp lệ trong MongoDB.")

    except Exception as e:
        print(f"❌ Lỗi khi kết nối hoặc truy vấn MongoDB: {e}")
        return

    # 2. Tải các file cache backup
    if not os.path.exists(OLD_EMB_PATH) or not os.path.exists(OLD_META_PATH):
        print(f"❌ Không tìm thấy file backup: '{OLD_EMB_PATH}' hoặc '{OLD_META_PATH}'.")
        return

    print(f"-> Đang tải file embedding gốc từ: {OLD_EMB_PATH}")
    old_embs = np.load(OLD_EMB_PATH)

    print(f"-> Đang tải file metadata gốc từ: {OLD_META_PATH}")
    with open(OLD_META_PATH, 'r', encoding='utf-8') as f:
        old_metas = json.load(f)

    if len(old_embs) != len(old_metas):
        print(f"⚠️ CẢNH BÁO: Số lượng embedding ({len(old_embs)}) và metadata ({len(old_metas)}) không khớp!")

    # 3. Tạo map để tra cứu nhanh embedding từ frame_id
    print("-> Lập bản đồ tra cứu (mapping) frame_id và embedding...")
    frame_id_to_emb_map = {meta['frame_id']: old_embs[i] for i, meta in enumerate(old_metas)}

    # 4. Lọc và tạo danh sách embedding mới
    print("-> Bắt đầu lọc embedding dựa trên dữ liệu từ MongoDB...")
    new_embs_list = []
    sorted_mongo_frame_ids = sorted(list(valid_frame_ids))

    for frame_id in tqdm(sorted_mongo_frame_ids, desc="Đang xử lý các frame"):
        if frame_id in frame_id_to_emb_map:
            new_embs_list.append(frame_id_to_emb_map[frame_id])
        else:
            print(f"-> Cảnh báo: không tìm thấy embedding cho frame_id '{frame_id}' trong file backup.")

    print(f"-> Hoàn tất lọc. Số lượng embedding được giữ lại: {len(new_embs_list)}")

    # 5. Lưu file .npy mới
    if not new_embs_list:
        print("❌ Không có embedding nào được giữ lại. Quá trình dừng lại.")
        return

    final_embs = np.array(new_embs_list)
    print(f"-> Đang lưu file embedding mới tại: {NEW_EMB_OUTPUT_PATH}")
    np.save(NEW_EMB_OUTPUT_PATH, final_embs)

    print("\n✅ Hoàn tất! File .npy đã được tạo và đồng bộ với collection MongoDB.")
    print(f"   - File mới: '{os.path.basename(NEW_EMB_OUTPUT_PATH)}'")
    print(f"   - Tổng số vector: {len(final_embs)}")

if __name__ == "__main__":
    # << 3. SỬ DỤNG asyncio.run() ĐỂ CHẠY HÀM ASYNC
    asyncio.run(sync_embeddings_from_mongo())