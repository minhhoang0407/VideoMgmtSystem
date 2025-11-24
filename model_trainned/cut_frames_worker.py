import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import open_clip
import tempfile
import requests
import asyncio
import cloudinary.uploader
from datetime import datetime, timezone
import json

# Thêm sys.path để import database connection
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import db

# ============================
# Config
# ============================
BASE_DIR = r"D:\KLCN\backend_app\assets"
EMBEDDINGS_NPY = os.path.join(BASE_DIR, "cached_frame_embs_RESYNCED.npy")
IDS_JSON = os.path.join(BASE_DIR, "frame_ids_RESYNCED.json")
KMEANS_DIR = os.path.join(BASE_DIR, "kmeans")

VIDEOS_COLLECTION = "videos"
FRAMES_COLLECTION = "frames"

API_URL = "http://127.0.0.1:8000" # Địa chỉ server API

SAMPLE_RATE = 1.0
K = 30
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(KMEANS_DIR, exist_ok=True)

# ============================
# Init CLIP model
# ============================
print("Initializing CLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai",
    quick_gelu=True
)
clip_model = clip_model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')
print("CLIP model loaded.")

# ============================
# Helper Functions
# ============================
def log(msg):
    print(f"[LOG] {msg}")

def download_video(video_url, save_path):
    """Tải video đồng bộ bằng requests"""
    try:
        print(f"📥 Downloading video from {video_url}...")
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Video saved: {save_path}")
        return save_path
    except Exception as e:
        log(f"❌ Error downloading video: {e}")
        return None

def encode_frames(paths, batch_size=BATCH_SIZE):
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch_size), desc="🤖 Encoding Frames"):
            batch = paths[i:i+batch_size]
            imgs = []
            for p in batch:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(clip_preprocess(img))
                except Exception as e:
                    log(f"⚠️ Error loading image {p}: {e}")
            if imgs:
                tensor = torch.stack(imgs).to(DEVICE)
                feats = clip_model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embs.append(feats.cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.empty((0, 768), dtype=np.float32)

def select_keyframes(frame_features, k=30):
    k = min(k, len(frame_features))
    if k <= 0:
        return [], None, None
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(frame_features)
    selected_indices = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue
        center = kmeans.cluster_centers_[i]
        points_in_cluster = frame_features[cluster_indices]
        dists = np.linalg.norm(points_in_cluster - center, axis=1)
        best_idx_in_cluster = np.argmin(dists)
        original_idx = cluster_indices[best_idx_in_cluster]
        selected_indices.append(original_idx)
    return sorted(selected_indices), labels, kmeans

def plot_kmeans(features, labels, video_id, kmeans=None, selected_indices=None, timestamps=None):
    if len(features) < 2:
        return
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=25, edgecolor='k', alpha=0.8)
    if kmeans:
        centers_2d = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='X', c='red', s=120, label='Cluster centers')
    if selected_indices:
        plt.scatter(reduced[selected_indices, 0], reduced[selected_indices, 1], s=140, facecolors='none', edgecolors='black', linewidths=2.5, label='Keyframes')
    plt.title(f"{video_id} | Total frames: {len(features)} | Clusters: {kmeans.n_clusters if kmeans else len(np.unique(labels))}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    os.makedirs(KMEANS_DIR, exist_ok=True)
    save_path = os.path.join(KMEANS_DIR, f"{video_id}_kmeans.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    log(f"📊 KMeans plot saved at {save_path}")

async def upload_to_cloud(local_path, file_name):
    """Upload ảnh lên Cloudinary"""
    def _upload():
        return cloudinary.uploader.upload(local_path, resource_type="image", folder="frames")
    try:
        result = await asyncio.to_thread(_upload)
        return result.get("secure_url")
    except Exception as e:
        log(f"❌ Error uploading {file_name}: {e}")
        return None

# ============================
# Core video processing
# ============================
async def process_new_video(video_doc):
    video_id = video_doc["_id"]
    video_url = video_doc["url"]
    video_title = video_doc.get("title", str(video_id))
    log(f"🎬 Processing video: {video_title} (ID: {video_id})")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_video_path = os.path.join(temp_dir, f"{video_title}.mp4")
            if not download_video(video_url, local_video_path):
                raise Exception("Cannot download video.")
            
            # --- 1. Trích xuất Frame ---
            cap = cv2.VideoCapture(local_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = int(fps * SAMPLE_RATE)
            temp_frame_paths, timestamps = [], []
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    temp_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
                    cv2.imwrite(temp_path, frame)
                    temp_frame_paths.append(temp_path)
                    timestamps.append(f"{frame_count/fps:.2f}s")
                frame_count += 1
            cap.release()
            
            if not temp_frame_paths:
                raise Exception("No frames extracted.")
            log(f"🎞️ {len(temp_frame_paths)} frames extracted.")

            # --- 2. Encode Frame ---
            all_frame_embs = encode_frames(temp_frame_paths)
            if all_frame_embs.shape[0] == 0:
                raise Exception("No frame embeddings generated.")

            # --- 3. Chọn Keyframe (KMeans) ---
            keyframe_indices, labels, kmeans_model = select_keyframes(all_frame_embs, k=K)
            log(f"✨ Selected {len(keyframe_indices)} keyframes.")

            # --- 4. Vẽ biểu đồ ---
            safe_filename = "".join(c for c in video_title if c.isalnum() or c in (' ')).rstrip().replace(' ','')
            plot_kmeans(all_frame_embs, labels, safe_filename, kmeans_model, keyframe_indices, timestamps)

            # --- 5. Upload lên Cloud ---
            cloud_urls = []
            for idx in keyframe_indices:
                local_path = temp_frame_paths[idx]
                file_name = os.path.basename(local_path)
                url = await upload_to_cloud(local_path, file_name)
                cloud_urls.append(url)

            # --- 6. Chuẩn bị dữ liệu lưu trữ ---
            keyframe_metadata_list = []
            keyframe_embeddings_list = []
            
            for i, idx in enumerate(keyframe_indices):
                cloud_url = cloud_urls[i]
                if cloud_url:
                    # Tạo Frame ID chuẩn: VideoID_SốThứTựFrame
                    frame_id = f"{video_id}_{idx:06d}"
                    
                    keyframe_metadata_list.append({
                        "frame_id": frame_id,
                        "video_id": video_id,
                        "path": cloud_url,
                        "timestamp": timestamps[idx],
                        "video_path": video_url,
                    })
                    keyframe_embeddings_list.append(all_frame_embs[idx])

        # --- Hết block 'with', file tạm đã xóa ---

        if keyframe_metadata_list:
            # A. Lưu vào MongoDB
            frames_collection = db[FRAMES_COLLECTION]
            await frames_collection.insert_many(keyframe_metadata_list)
            log(f"💾 Inserted {len(keyframe_metadata_list)} keyframes into MongoDB.")

            # B. Cập nhật file .npy (Vector) - APPEND
            new_embs = np.array(keyframe_embeddings_list, dtype='float32')
            if os.path.exists(EMBEDDINGS_NPY):
                try:
                    old_embs = np.load(EMBEDDINGS_NPY)
                    all_embs = np.concatenate([old_embs, new_embs], axis=0)
                except Exception as e:
                    log(f"⚠️ File npy lỗi, tạo mới: {e}")
                    all_embs = new_embs
            else:
                all_embs = new_embs
            np.save(EMBEDDINGS_NPY, all_embs)

            # C. Cập nhật file .json (IDs) - APPEND
            new_ids = [item['frame_id'] for item in keyframe_metadata_list]
            current_ids = []
            if os.path.exists(IDS_JSON):
                try:
                    with open(IDS_JSON, 'r') as f:
                        current_ids = json.load(f)
                except Exception as e:
                    log(f"⚠️ File json lỗi, tạo mới: {e}")
                    current_ids = []
            
            current_ids.extend(new_ids) # Nối đuôi
            
            with open(IDS_JSON, 'w') as f:
                json.dump(current_ids, f)

            log(f"✅ Đã cập nhật NPY & JSON. Tổng cộng: {len(all_embs)} vectors.")

            # D. Kiểm tra an toàn
            if len(all_embs) != len(current_ids):
                log("❌ CẢNH BÁO: Số lượng Vector và ID bị lệch! Cần chạy resync.")

        # 7. Đánh dấu hoàn tất
        videos_collection = db[VIDEOS_COLLECTION]
        await videos_collection.update_one(
            {"_id": video_id},
            {"$set": {
                "status_video": "COMPLETED", 
                "processed_at": datetime.now(timezone.utc)
            }}
        )
        log(f"🏁 Video {video_id} processing completed.")

        # ---GỌI API RELOAD ---
        try:
            log("🔄 Triggering API Server to reload index...")
            # Gọi API reload
            # Lưu ý: Cần chạy trong thread riêng hoặc dùng thư viện requests (sync) vì worker loop đang async
            response = requests.post(f"{API_URL}/frames/reload-index", timeout=5)
            if response.status_code == 200:
                log(f"✅ API Reloaded: {response.json()}")
            else:
                log(f"⚠️ API Reload Warning: {response.text}")
        except Exception as e:
            log(f"❌ Failed to trigger API reload (Server might be down): {e}")

    except Exception as e:
        log(f"❌ Error processing video {video_id}: {e}")
        videos_collection = db[VIDEOS_COLLECTION]
        await videos_collection.update_one(
            {"_id": video_id},
            {"$set": {"status_video": "FAILED", "error_message": str(e)}}
        )

# ============================
# Worker loop (sequential)
# ============================
async def main_worker_loop():
    log("🚀 Worker started (sequential mode).")
    while True:
        pending_video = await db[VIDEOS_COLLECTION].find_one({"status_video": "PENDING"})
        if pending_video:
            await process_new_video(pending_video)
        else:
            log("⏸ No pending videos. Sleeping 30s...")
            await asyncio.sleep(30)

# ============================
# Run worker
# ============================
if __name__ == "__main__":
    asyncio.run(main_worker_loop())
