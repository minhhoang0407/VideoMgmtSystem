# import os
# import cv2
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from sklearn.cluster import KMeans
# import torch
# import open_clip
# import tempfile
# import requests
# import asyncio
# import cloudinary.uploader
# from datetime import datetime, timezone
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# # Thêm sys.path để import database connection
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from database.connection import db
# # ============================
# # Config
# # ============================
# BASE_DIR = r"D:\KLCN\backend_app\assets"
# EMBEDDINGS_NPY = os.path.join(BASE_DIR, "cached_frame_embs_synced.npy")
# KMEANS_DIR = os.path.join(BASE_DIR, "kmeans") # Thư mục để lưu biểu đồ
# # Tên các collection trong MongoDB
# VIDEOS_COLLECTION = "videos"
# FRAMES_COLLECTION = "frames"

# # Các tham số xử lý
# SAMPLE_RATE = 1
# K = 30
# BATCH_SIZE = 16
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # ============================
# # Init CLIP model
# # ============================
# print("Initializing CLIP model...")
# clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#     "ViT-L-14",
#     pretrained="openai",
#     quick_gelu=True  # Add this parameter
# )
# clip_model = clip_model.to(DEVICE).eval()
# print("CLIP model loaded.")
# # ============================
# # Cloud & Helper Functions
# # (Các hàm này không thay đổi)
# # ============================
# def log(msg):
#     print(f"[LOG] {msg}")

# def download_video_from_url(video_url, save_path):
#     # ... (Giữ nguyên logic download)
#     try:
#         print(f"📥 Đang tải video từ {video_url}...")
#         with requests.get(video_url, stream=True, timeout=60) as r:
#             r.raise_for_status()
#             with open(save_path, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
#         print(f"✅ Tải video thành công: {save_path}")
#         return save_path
#     except Exception as e:
#         print(f"❌ Lỗi khi tải video {video_url}: {e}")
#         return None

# async def upload_to_cloud(local_path, file_name):
#     # ... (Giữ nguyên logic upload)
#     def _upload():
#         return cloudinary.uploader.upload(local_path, resource_type="image", folder="frames")
#     try:
#         result = await asyncio.to_thread(_upload)
#         return result.get("secure_url")
#     except Exception as e:
#         print(f"❌ Lỗi khi upload {file_name} lên Cloudinary: {e}")
#         return None

# # ============================
# # KMeans Visualization
# # ============================
# def plot_kmeans(features, labels, video_id, kmeans=None, selected_indices=None, timestamps=None):
#     if len(features) < 2:
#         return

#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(features)

#     h = .02
#     x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
#     y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#     if kmeans is not None:
#         Z = kmeans.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
#         Z = Z.reshape(xx.shape)
#         plt.contourf(xx, yy, Z, cmap="tab20", alpha=0.15)

#     plt.scatter(reduced[:, 0], reduced[:, 1], c=labels,
#                 s=25, cmap="tab20", edgecolor='k', alpha=0.8)

#     if kmeans is not None:
#         centers_2d = pca.transform(kmeans.cluster_centers_)
#         plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
#                     marker='X', c='red', s=120, label='Cluster centers')

#     for i in range(kmeans.n_clusters if kmeans else len(np.unique(labels))):
#         points = reduced[labels == i]
#         if len(points) > 2:
#             cov = np.cov(points, rowvar=False)
#             vals, vecs = np.linalg.eigh(cov)
#             order = vals.argsort()[::-1]
#             vals, vecs = vals[order], vecs[:, order]
#             theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
#             width, height = 2.5 * np.sqrt(vals)
#             ellip = plt.matplotlib.patches.Ellipse(xy=np.mean(points, axis=0),
#                                                    width=width, height=height,
#                                                    angle=theta,
#                                                    edgecolor='gray', facecolor='none',
#                                                    linestyle='--', linewidth=1.2, alpha=0.8)
#             plt.gca().add_artist(ellip)
#             cx, cy = np.mean(points, axis=0)
#             plt.text(cx, cy, f"{i+1}", fontsize=9, color='darkred',
#                      ha='center', va='center', weight='bold', alpha=0.9)

#     if selected_indices is not None and len(selected_indices) > 0:
#         plt.scatter(reduced[selected_indices, 0],
#                     reduced[selected_indices, 1],
#                     s=140, facecolors='none', edgecolors='black',
#                     linewidths=2.5, label='Keyframes')
#         if timestamps is not None:
#             for idx in selected_indices:
#                 if idx < len(timestamps):
#                     ts = timestamps[idx]
#                     x, y = reduced[idx, 0], reduced[idx, 1]
#                     plt.text(x + 0.05, y + 0.05, ts,
#                              fontsize=7, color='darkblue',
#                              weight='bold', alpha=0.9)

#     k_val = kmeans.n_clusters if kmeans else len(np.unique(labels))
#     n_keyframes = len(selected_indices) if selected_indices is not None else 0
#     plt.title(
#         f"{video_id}\nTổng frame: {len(features)} | Cụm: {k_val} | Frame sau gom: {n_keyframes}",
#         fontsize=11, weight='bold', color='darkblue'
#     )
#     plt.legend(loc='best', fontsize=8)
#     plt.xlabel("PCA Dimension 1")
#     plt.ylabel("PCA Dimension 2")
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.tight_layout()

#     os.makedirs(KMEANS_DIR, exist_ok=True)
#     save_path = os.path.join(KMEANS_DIR, f"{video_id}_kmeans.png")
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     log(f"📊 Đã lưu biểu đồ KMeans cho {video_id} tại {save_path}")
# # ============================
# # Chọn keyframe bằng KMeans
# # ============================
# def select_keyframes(frame_features, k=30):
#     k = min(k, len(frame_features))
#     if k <= 0:
#         return [], None, None # Trả về None nếu không xử lý

#     kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
#     labels = kmeans.fit_predict(frame_features)
    
#     selected_indices = []
#     for i in range(k):
#         cluster_indices = np.where(labels == i)[0]
#         if len(cluster_indices) == 0:
#             continue
#         center = kmeans.cluster_centers_[i]
#         points_in_cluster = frame_features[cluster_indices]
#         dists = np.linalg.norm(points_in_cluster - center, axis=1)
#         best_idx_in_cluster = np.argmin(dists)
#         original_idx = cluster_indices[best_idx_in_cluster]
#         selected_indices.append(original_idx)
        
#     # Trả về thêm 'labels' và 'kmeans' để dùng cho việc vẽ biểu đồ
#     return sorted(selected_indices), labels, kmeans

# def encode_frames(paths, batch_size=BATCH_SIZE):
#     embs = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(paths), batch_size), desc="🤖 Encoding Frames"):
#             batch = paths[i:i+batch_size]
#             imgs = []
#             for p in batch:
#                 try:
#                     img = Image.open(p).convert("RGB")
#                     imgs.append(clip_preprocess(img))
#                 except Exception as e:
#                     log(f"⚠️ Lỗi load ảnh {p}: {e}")
#             if imgs:
#                 tensor = torch.stack(imgs).to(DEVICE)
#                 feats = clip_model.encode_image(tensor)
#                 feats = feats / feats.norm(dim=-1, keepdim=True)
#                 embs.append(feats.cpu().numpy())
#     return np.concatenate(embs, axis=0) if embs else np.empty((0, 768), dtype=np.float32)

# # ============================
# # << CORE LOGIC >> Pipeline xử lý chính cho một video
# # ============================
# async def process_new_video(video_doc: dict):
#     """
#     Hàm xử lý hoàn chỉnh cho một video: tải về, phân tích, và cập nhật kết quả.
#     """
#     video_id = video_doc["_id"]
#     video_url = video_doc["url"]
#     video_title = video_doc["title"]
#     log(f"🎬 Bắt đầu xử lý video: {video_title} (ID: {video_id})")

#     try:
#         # Sử dụng thư mục tạm để đảm bảo các file được dọn dẹp sạch sẽ
#         with tempfile.TemporaryDirectory() as temp_dir:
#             local_video_path = os.path.join(temp_dir, f"{video_title}.mp4")

#             # 1. Tải video từ cloud về máy tạm
#             if not download_video_from_url(video_url, local_video_path):
#                 raise Exception("Không thể tải video từ cloud.")

#             # 2. Cắt video thành các frame thô (logic từ file gốc)
#             cap = cv2.VideoCapture(local_video_path)
#             fps = cap.get(cv2.CAP_PROP_FPS) or 30
#             frame_interval = int(fps * SAMPLE_RATE)
            
#             temp_frame_paths, timestamps = [], []
#             frame_count = 0
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 if frame_count % frame_interval == 0:
#                     temp_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.jpg")
#                     cv2.imwrite(temp_path, frame)
#                     temp_frame_paths.append(temp_path)
#                     timestamps.append(f"{frame_count / fps:.2f}s")
#                 frame_count += 1
#             cap.release()
            
#             if not temp_frame_paths:
#                 log(f"⚠️ Không trích xuất được frame nào từ {video_id}.")
#                 raise Exception("Không trích xuất được frame.")
#             log(f"🎞️  Trích xuất được {len(temp_frame_paths)} frame thô.")

#             # 3. Encode tất cả frame thô bằng CLIP
#             all_frame_embs = encode_frames(temp_frame_paths)
#             if all_frame_embs.shape[0] == 0:
#                 raise Exception("Không encode được frame nào.")

#             # 4. Dùng K-Means để chọn ra các keyframe
#             keyframe_indices, labels, kmeans_model = select_keyframes(all_frame_embs, k=K) # Sửa ở đây
#             log(f"✨ Chọn được {len(keyframe_indices)} keyframes bằng K-Means.")

#             # Thêm ngay sau đó:
#             # 4.1. Vẽ và lưu biểu đồ K-Means
#             if kmeans_model: # Chỉ vẽ nếu K-Means chạy thành công
#                 # Làm sạch title để dùng làm tên file an toàn
#                 safe_filename = "".join(c for c in video_title if c.isalnum() or c in (' ')).rstrip()
#                 safe_filename = safe_filename.replace(' ')

#                 plot_kmeans(
#                     features=all_frame_embs,
#                     labels=labels,
#                     video_id=safe_filename, # << SỬ DỤNG TITLE Ở ĐÂY
#                     kmeans=kmeans_model,
#                     selected_indices=keyframe_indices,
#                     timestamps=timestamps
#                 )

#             # 5. Upload các keyframe đã chọn lên cloud
#             upload_tasks = []
#             for idx in keyframe_indices:
#                 local_path = temp_frame_paths[idx]
#                 file_name = os.path.basename(local_path)
#                 upload_tasks.append(upload_to_cloud(local_path, file_name))
            
#             # Chạy các tác vụ upload song song để tăng tốc độ
#             cloud_urls = await asyncio.gather(*upload_tasks)

#             keyframe_metadata_list = []
#             keyframe_embeddings_list = []
#             for i, idx in enumerate(keyframe_indices):
#                 cloud_url = cloud_urls[i]
#                 if cloud_url:
#                     keyframe_metadata_list.append({
#                         "frame_id": f"{video_id}_{idx:06d}",
#                         "video_id": video_id,
#                         "path": cloud_url,
#                         "timestamp": timestamps[idx],
#                         "video_path": video_url,
#                     })
#                     keyframe_embeddings_list.append(all_frame_embs[idx])

#         # Hết khối `with`, thư mục tạm và các file bên trong sẽ tự động bị xóa

#         # 6. Cập nhật MongoDB với các keyframe mới
#         if keyframe_metadata_list:
#             log(f"💾 Đang ghi {len(keyframe_metadata_list)} keyframes vào MongoDB...")
#             frames_collection = db[FRAMES_COLLECTION]
#             await frames_collection.insert_many(keyframe_metadata_list)
#             log("✅ Ghi keyframes vào MongoDB thành công.")

#         # 7. Cập nhật file embedding .npy
#         if keyframe_embeddings_list:
#             new_embs = np.array(keyframe_embeddings_list)
#             log(f"💾 Đang cập nhật file embedding: {EMBEDDINGS_NPY}")
#             if os.path.exists(EMBEDDINGS_NPY):
#                 old_embs = np.load(EMBEDDINGS_NPY)
#                 all_embs = np.concatenate([old_embs, new_embs], axis=0)
#             else:
#                 all_embs = new_embs
#             np.save(EMBEDDINGS_NPY, all_embs)
#             log(f"✅ Cập nhật file embedding thành công, tổng số vector: {len(all_embs)}")

#         # 8. Đánh dấu video đã xử lý thành công
#         videos_collection = db[VIDEOS_COLLECTION]
#         await videos_collection.update_one(
#             {"_id": video_id},
#             {"$set": {
#                 "status_video": "COMPLETED",
#                 "processed_at": datetime.now(timezone.utc)
#             }}
#         )
#         log(f"🏁 Hoàn tất xử lý video: {video_id}")

#     except Exception as e:
#         log(f"❌ Lỗi nghiêm trọng khi xử lý {video_id}: {e}")
#         # Đánh dấu video xử lý thất bại
#         videos_collection = db[VIDEOS_COLLECTION]
#         await videos_collection.update_one(
#             {"_id": video_id},
#             {"$set": {
#                 "status_video": "FAILED",
#                 "error_message": str(e)
#             }}
#         )
# # ============================
# # << WORKER LOOP >> Vòng lặp chính của worker
# # ============================
# async def main_worker_loop():
#     log("🚀 Worker xử lý video đã khởi động, đang tìm kiếm các video PENDING...")
#     videos_collection = db[VIDEOS_COLLECTION]
    
#     while True:
#         try:
#             # Tìm một video PENDING và cập nhật ngay thành PROCESSING
#             # Thao tác này đảm bảo không worker nào khác lấy cùng video
#             pending_video = await videos_collection.find_one_and_update(
#                 {"status_video": "PENDING"},
#                 {"$set": {"status_video": "PROCESSING", "processed_at": datetime.now(timezone.utc)}}
#             )

#             if pending_video:
#                 # Nếu tìm thấy, bắt đầu xử lý
#                 await process_new_video(pending_video)
#             else:
#                 # Nếu không có video nào, đợi 30 giây rồi tìm lại
#                 log("...Không có video mới. Tạm nghỉ 30 giây...")
#                 await asyncio.sleep(30)

#         except Exception as e:
#             log(f"🔥 Lỗi trong vòng lặp chính của worker: {e}")
#             await asyncio.sleep(15) # Đợi lâu hơn nếu có lỗi nghiêm trọng

# #main            
# #if __name__ == "__main__":
# #    asyncio.run(main_worker_loop())

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

# Thêm sys.path để import database connection
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.connection import db

# ============================
# Config
# ============================
BASE_DIR = r"D:\KLCN\backend_app\assets"
EMBEDDINGS_NPY = os.path.join(BASE_DIR, "cached_frame_embs_RESYNCED.npy")
KMEANS_DIR = os.path.join(BASE_DIR, "kmeans")

VIDEOS_COLLECTION = "videos"
FRAMES_COLLECTION = "frames"

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
            # Extract frames
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

            # Encode frames
            all_frame_embs = encode_frames(temp_frame_paths)
            if all_frame_embs.shape[0] == 0:
                raise Exception("No frame embeddings generated.")

            # KMeans keyframes
            keyframe_indices, labels, kmeans_model = select_keyframes(all_frame_embs, k=K)
            log(f"✨ Selected {len(keyframe_indices)} keyframes.")

            # Plot KMeans
            safe_filename = "".join(c for c in video_title if c.isalnum() or c in (' ')).rstrip().replace(' ','')
            plot_kmeans(all_frame_embs, labels, safe_filename, kmeans_model, keyframe_indices, timestamps)

            # Upload keyframes
            cloud_urls = []
            for idx in keyframe_indices:
                local_path = temp_frame_paths[idx]
                file_name = os.path.basename(local_path)
                url = await upload_to_cloud(local_path, file_name)
                cloud_urls.append(url)

            # Prepare MongoDB data
            keyframe_metadata_list = []
            keyframe_embeddings_list = []
            for i, idx in enumerate(keyframe_indices):
                cloud_url = cloud_urls[i]
                if cloud_url:
                    keyframe_metadata_list.append({
                        "frame_id": f"{video_id}_{idx:06d}",
                        "video_id": video_id,
                        "path": cloud_url,
                        "timestamp": timestamps[idx],
                        "video_path": video_url,
                    })
                    keyframe_embeddings_list.append(all_frame_embs[idx])

        # Insert frames into MongoDB
        if keyframe_metadata_list:
            frames_collection = db[FRAMES_COLLECTION]
            await frames_collection.insert_many(keyframe_metadata_list)
            log(f"💾 Inserted {len(keyframe_metadata_list)} keyframes into MongoDB.")

        # Update embeddings .npy
        if keyframe_embeddings_list:
            new_embs = np.array(keyframe_embeddings_list)
            if os.path.exists(EMBEDDINGS_NPY):
                old_embs = np.load(EMBEDDINGS_NPY)
                all_embs = np.concatenate([old_embs, new_embs], axis=0)
            else:
                all_embs = new_embs
            np.save(EMBEDDINGS_NPY, all_embs)
            log(f"💾 Updated embeddings file. Total vectors: {len(all_embs)}")

        # Mark video completed
        videos_collection = db[VIDEOS_COLLECTION]
        await videos_collection.update_one(
            {"_id": video_id},
            {"$set": {"status_video": "COMPLETED", "processed_at": datetime.now(timezone.utc)}}
        )
        log(f"🏁 Video {video_id} processing completed.")

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
