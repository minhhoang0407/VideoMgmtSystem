import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import open_clip

# ============================
# Config
# ============================
BASE_DIR = r"D:\KLCN\backend_app\model_trainned"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
KMEANS_DIR = os.path.join(BASE_DIR, "kmeans")

FRAMES_JSONL = os.path.join(BASE_DIR, "frames.jsonl")
CACHE_FRAME_EMB = os.path.join(BASE_DIR, "cached_frame_embs.npy")
CACHE_FRAME_META = os.path.join(BASE_DIR, "cached_frame_meta.json")

SAMPLE_RATE = 1.0  # 1 frame mỗi giây
K = 30              # số keyframes
BATCH_SIZE = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Init CLIP model
# ============================
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
clip_model = clip_model.to(DEVICE).eval()

# ============================
# Helper
# ============================
def log(msg):
    print(f"[LOG] {msg}")

def save_cache(embs, metas):
    np.save(CACHE_FRAME_EMB, embs)
    np.save(CACHE_FRAME_META, np.array(metas, dtype=object))

def load_cache():
    if not os.path.exists(CACHE_FRAME_EMB) or not os.path.exists(CACHE_FRAME_META):
        return None, None
    return np.load(CACHE_FRAME_EMB, allow_pickle=True), np.load(CACHE_FRAME_META, allow_pickle=True).tolist()

# ============================
# KMeans Visualization
# ============================
def plot_kmeans(features, labels, video_id, kmeans=None, selected_indices=None, timestamps=None):
    if len(features) < 2:
        return

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    h = .02
    x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
    y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if kmeans is not None:
        Z = kmeans.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap="tab20", alpha=0.15)

    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels,
                s=25, cmap="tab20", edgecolor='k', alpha=0.8)

    if kmeans is not None:
        centers_2d = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
                    marker='X', c='red', s=120, label='Cluster centers')

    for i in range(kmeans.n_clusters if kmeans else len(np.unique(labels))):
        points = reduced[labels == i]
        if len(points) > 2:
            cov = np.cov(points, rowvar=False)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2.5 * np.sqrt(vals)
            ellip = plt.matplotlib.patches.Ellipse(xy=np.mean(points, axis=0),
                                                   width=width, height=height,
                                                   angle=theta,
                                                   edgecolor='gray', facecolor='none',
                                                   linestyle='--', linewidth=1.2, alpha=0.8)
            plt.gca().add_artist(ellip)
            cx, cy = np.mean(points, axis=0)
            plt.text(cx, cy, f"{i+1}", fontsize=9, color='darkred',
                     ha='center', va='center', weight='bold', alpha=0.9)

    if selected_indices is not None and len(selected_indices) > 0:
        plt.scatter(reduced[selected_indices, 0],
                    reduced[selected_indices, 1],
                    s=140, facecolors='none', edgecolors='black',
                    linewidths=2.5, label='Keyframes')
        if timestamps is not None:
            for idx in selected_indices:
                if idx < len(timestamps):
                    ts = timestamps[idx]
                    x, y = reduced[idx, 0], reduced[idx, 1]
                    plt.text(x + 0.05, y + 0.05, ts,
                             fontsize=7, color='darkblue',
                             weight='bold', alpha=0.9)

    k_val = kmeans.n_clusters if kmeans else len(np.unique(labels))
    n_keyframes = len(selected_indices) if selected_indices is not None else 0
    plt.title(
        f"{video_id}\nTổng frame: {len(features)} | Cụm: {k_val} | Frame sau gom: {n_keyframes}",
        fontsize=11, weight='bold', color='darkblue'
    )
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs(KMEANS_DIR, exist_ok=True)
    save_path = os.path.join(KMEANS_DIR, f"{video_id}_kmeans.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

# ============================
# Chọn keyframe bằng KMeans
# ============================
def select_keyframes(frame_features, k=30):
    k = min(k, len(frame_features))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(frame_features)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    selected_indices = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        center = centers[i]
        dists = [np.linalg.norm(frame_features[j] - center) for j in cluster_indices]
        best_idx = cluster_indices[np.argmin(dists)]
        selected_indices.append(best_idx)
    return selected_indices, labels, kmeans

# ============================
# Encode frames bằng CLIP
# ============================
def encode_frames(paths, batch_size=BATCH_SIZE):
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            imgs = []
            for p in batch:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(clip_preprocess(img))
                except Exception as e:
                    log(f"⚠️ Lỗi load ảnh {p}: {e}")
            if imgs:
                tensor = torch.stack(imgs).to(DEVICE)
                feats = clip_model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embs.append(feats.cpu().numpy())
    return np.concatenate(embs, axis=0) if embs else np.empty((0, 768), dtype=np.float32)

# ============================
# Cắt keyframe và lưu metadata
# ============================
def extract_keyframes(video_path, k=K, interval_sec=SAMPLE_RATE):
    from pathlib import Path
    video_id = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"⚠️ Không thể mở video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval_sec)
    frames, timestamps = [], []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
            timestamps.append(f"{idx / fps:.2f}s")
        idx += 1
    cap.release()

    if not frames:
        log(f"⚠️ Video {video_id} không có frame nào để xử lý.")
        return []

    os.makedirs(FRAME_DIR, exist_ok=True)
    frame_entries = []
    for i, f in enumerate(frames, start=1):
        frame_name = f"{video_id}_f{i:05d}.jpg"
        frame_path = os.path.join(FRAME_DIR, frame_name)
        cv2.imwrite(frame_path, f)
        frame_entries.append({
            "frame_id": f"{video_id}_{i:05d}",
            "video_id": video_id,
            "frame_path": frame_path.replace("\\", "/"),
            "timestamp": timestamps[i-1],
            "video_path": video_path.replace("\\", "/"),
        })

    all_embs, all_metas = update_frame_cache(frame_entries)
    if all_embs is None or len(all_embs) == 0:
        log(f"⚠️ Không có embedding hợp lệ cho video {video_id}.")
        return []

    vid_embs = []
    vid_indices = []
    for i, meta in enumerate(all_metas):
        if meta["video_id"] == video_id:
            vid_embs.append(all_embs[i])
            vid_indices.append(i)
    if not vid_embs:
        log(f"⚠️ Không tìm thấy embedding cho {video_id}.")
        return []

    features = np.vstack(vid_embs)
    k = min(k, len(features))
    if k <= 1:
        log(f"⚠️ Video {video_id} có quá ít frame ({len(features)}).")
        return []

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    selected_indices = []
    for i in range(k):
        cluster_points = features[labels == i]
        cluster_ids = np.where(labels == i)[0]
        center = kmeans.cluster_centers_[i]
        dists = np.linalg.norm(cluster_points - center, axis=1)
        best_idx = cluster_ids[np.argmin(dists)]
        selected_indices.append(best_idx)
    selected_indices = sorted(selected_indices)

    plot_kmeans(features, labels, video_id, kmeans=kmeans,
                selected_indices=selected_indices, timestamps=timestamps)

    selected_frames = [frame_entries[i] for i in selected_indices if i < len(frame_entries)]
    log(f"✅ Trích xuất {len(selected_frames)} keyframes cho {video_id} (tổng {len(frames)} frames).")
    return selected_frames

# ============================
# Encode & cập nhật cache
# ============================
def update_frame_cache(new_frames):
    old_embs, old_metas = load_cache()
    old_ids = {m["frame_id"] for m in old_metas} if old_metas else set()
    fresh_frames = [f for f in new_frames if f["frame_id"] not in old_ids]

    if not fresh_frames:
        log("✅ Không có frame mới cần encode.")
        return old_embs, old_metas

    log(f"🆕 Có {len(fresh_frames)} frame mới cần encode.")
    paths = [f["frame_path"] for f in fresh_frames]
    new_embs = encode_frames(paths)

    all_embs = np.concatenate([old_embs, new_embs], axis=0) if old_embs is not None else new_embs
    all_metas = old_metas + fresh_frames if old_metas else fresh_frames

    save_cache(all_embs, all_metas)
    log(f"✅ Cập nhật cache thành công, tổng số frame: {len(all_metas)}")
    return all_embs, all_metas

# ============================
# Pipeline chính
# ============================
def process_new_video(video_path):
    log(f"🎬 Đang xử lý video mới: {video_path}")
    frames = extract_keyframes(video_path)
    if not frames:
        log("⚠️ Không có keyframes được trích xuất.")
        return

    with open(FRAMES_JSONL, "a", encoding="utf-8") as f:
        for i, entry in enumerate(frames, start=1):
            json_entry = {
                "frame_id": f"{entry['video_id']}_{i:05d}",
                "video_id": entry["video_id"],
                "frame_path": entry["frame_path"].replace("\\", "/"),
                "timestamp": entry["timestamp"],
                "video_path": entry["video_path"].replace("\\", "/")
            }
            f.write(json.dumps(json_entry, ensure_ascii=False) + "\n")
    log(f"💾 Ghi {len(frames)} keyframes vào {FRAMES_JSONL}")

    update_frame_cache(frames)
    log(f"🏁 Hoàn tất xử lý video: {video_path}")

# ============================
# Quét video mới
# ============================
def get_processed_video_ids():
    if not os.path.exists(FRAMES_JSONL):
        return set()
    processed = set()
    with open(FRAMES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                processed.add(entry["video_id"])
            except:
                continue
    return processed

def scan_and_process_new_videos():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(KMEANS_DIR, exist_ok=True)

    all_videos = [
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    processed_ids = get_processed_video_ids()
    new_videos = [
        v for v in all_videos
        if os.path.splitext(os.path.basename(v))[0] not in processed_ids
    ]

    if not new_videos:
        log("✅ Không có video mới cần xử lý.")
        return

    log(f"🆕 Phát hiện {len(new_videos)} video mới cần xử lý.")
    for video_path in new_videos:
        try:
            process_new_video(video_path)
        except Exception as e:
            log(f"❌ Lỗi khi xử lý {video_path}: {e}")

# ============================
# Main
# ============================
if __name__ == "__main__":
    scan_and_process_new_videos()
