import os
import json
import time
import numpy as np
import faiss
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from googletrans import Translator  # ✅ thêm thư viện dịch

# ============================
# Config
# ============================
BASE_DIR = r"E:\DoAnKhoaLuan\model"
FRAME_DIR = os.path.join(BASE_DIR, "frames")
CACHE_FRAME_EMB = os.path.join(BASE_DIR, "cached_frame_embs.npy")
CACHE_FRAME_META = os.path.join(BASE_DIR, "cached_frame_meta.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
TOP_K = 20

# ============================
# Init CLIP model
# ============================
clip_model, clip_preprocess, _ = create_model_and_transforms("ViT-L-14", pretrained="openai")
clip_model = clip_model.to(DEVICE).eval()
clip_tokenizer = get_tokenizer("ViT-L-14")

# ============================
# Init BLIP-2 for caption
# ============================
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(DEVICE)

# ============================
# Translator init
# ============================
translator = Translator()

# ============================
# Helper log & cache
# ============================
def log(msg):
    print(f"[LOG] {msg}")

def save_cache(embs, metas):
    np.save(CACHE_FRAME_EMB, embs)
    with open(CACHE_FRAME_META, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

def load_cache():
    if not os.path.exists(CACHE_FRAME_EMB) or not os.path.exists(CACHE_FRAME_META):
        return None, None
    embs = np.load(CACHE_FRAME_EMB, allow_pickle=True)
    with open(CACHE_FRAME_META, "r", encoding="utf-8") as f:
        metas = json.load(f)
    return embs, metas

# ============================
# Dịch tiếng Việt sang tiếng Anh
# ============================
def translate_vi_to_en(text):
    """
    Dịch tiếng Việt sang tiếng Anh bằng Google Translate.
    Nếu lỗi, trả lại text gốc.
    """
    try:
        result = translator.translate(text, src="vi", dest="en")
        translated = result.text.strip()
        if translated.lower() != text.lower():
            log(f"🌐 Dịch sang tiếng Anh: '{translated}'")
        return translated
    except Exception as e:
        log(f"⚠️ Lỗi khi dịch: {e}")
        return text

# ============================
# Encode query image(s)
# ============================
def encode_query_images(image_paths):
    all_feats = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(clip_preprocess(img))
            except Exception as e:
                log(f"⚠️ Lỗi load ảnh {p}: {e}")
        if imgs:
            tensor = torch.stack(imgs).to(DEVICE)
            with torch.no_grad():
                feats = clip_model.encode_image(tensor)
                feats /= feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 768), dtype=np.float32)

# ============================
# Encode text query
# ============================
def encode_text_query(text):
    tokens = clip_tokenizer([text])
    tokens = tokens.to(DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()

# ============================
# BLIP-2 caption generator
# ============================
def generate_caption(image_path):
    img = Image.open(image_path).convert("RGB")
    inputs = blip2_processor(images=img, return_tensors="pt").to(DEVICE)
    out = blip2_model.generate(**inputs)
    caption = blip2_processor.decode(out[0], skip_special_tokens=True)
    return caption

# ============================
# Build / update FAISS
# ============================
def build_faiss_index(embs):
    if embs is None or len(embs) == 0:
        return None
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embs)
    index.add(embs)
    log(f"✅ FAISS index ready: {index.ntotal} vectors, dim={dim}")
    return index

# ============================
# Search top-k
# ============================
def search_topk(index, metas, query_emb, top_k=TOP_K):
    if index is None or query_emb is None or len(query_emb) == 0:
        log("❌ Không có index hoặc query embedding.")
        return []

    faiss.normalize_L2(query_emb)
    D, I = index.search(query_emb, top_k * 5)

    results = []
    for q_idx, ids in enumerate(I):
        seen_videos = {}
        for rank, idx in enumerate(ids):
            if idx >= len(metas):
                continue
            meta = metas[idx]
            vid = meta.get("video_id")
            score = float(D[q_idx][rank])
            if vid not in seen_videos or score > seen_videos[vid][0]:
                seen_videos[vid] = (score, {
                    "rank": rank + 1,
                    "score": score,
                    "video_id": vid,
                    "frame_id": meta.get("frame_id"),
                    "path": meta.get("path"),
                    "timestamp": meta.get("timestamp"),
                })
        filtered = sorted([v[1] for v in seen_videos.values()],
                          key=lambda x: x["score"], reverse=True)[:top_k]
        results.append(filtered)
    return results

# ============================
# Encode một ảnh trực tiếp (CLIP)
# ============================
def encode_query_image_only(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_image(tensor)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()
    except Exception as e:
        log(f"⚠️ Lỗi encode ảnh {image_path}: {e}")
        return np.empty((1, 768), dtype=np.float32)

# ============================
# Encode caption sinh bởi BLIP-2
# ============================
def encode_query_caption(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = blip2_processor(images=img, return_tensors="pt").to(DEVICE)
        out = blip2_model.generate(**inputs)
        caption = blip2_processor.decode(out[0], skip_special_tokens=True)
        tokens = clip_tokenizer([caption]).to(DEVICE)
        with torch.no_grad():
            feat = clip_model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy(), caption
    except Exception as e:
        log(f"⚠️ Lỗi encode caption cho ảnh {image_path}: {e}")
        return np.empty((1, 768), dtype=np.float32), ""

# ============================
# Kết hợp 2 embedding
# ============================
def combine_embeddings(img_emb, text_emb, alpha=0.7):
    if img_emb.shape != text_emb.shape:
        log("⚠️ Kích thước embedding khác nhau, sử dụng image embedding thuần")
        return img_emb
    combined = alpha * img_emb + (1 - alpha) * text_emb
    return combined

# ============================
# Chuẩn hóa embedding
# ============================
def normalize_emb(emb):
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norm + 1e-10)

# ============================
# Hiển thị frame tốt nhất
# ============================
def show_best_frame(results):
    if not results or len(results[0]) == 0:
        log("🔍 Không có frame để hiển thị.")
        return
    best = results[0][0]
    frame_path = best["path"]
    score = best["score"]
    try:
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((10, 10), f"Score: {score:.4f}", fill="red", font=font)
        img.show(title=f"Best Frame (Score={score:.4f})")
    except Exception as e:
        log(f"⚠️ Lỗi mở frame {frame_path}: {e}")

# ============================
# Main pipeline interactive
# ============================
def main():
    log("\n🚀 CLIP + BLIP-2 Image+Text Retrieval Pipeline (Final)\n")

    embs, metas = load_cache()
    if embs is None or metas is None or len(metas) == 0:
        log("❌ Chưa có cache embedding frame nào.")
        return

    index = build_faiss_index(embs)

    qtype = input("Nhập loại query ('text' hoặc 'image'): ").strip().lower()
    if qtype == "text":
        query_text = input("📝 Nhập câu query (có thể là tiếng Việt): ").strip()
        translated_text = translate_vi_to_en(query_text)
        t0 = time.time()
        query_emb = encode_text_query(translated_text)
        log(f"⏱️ Encode text query xong: {time.time() - t0:.2f}s")

    elif qtype == "image":
        image_path = input("🖼️ Nhập đường dẫn ảnh query: ").strip()
        if not os.path.exists(image_path):
            log("❌ Ảnh không tồn tại.")
            return
        t0 = time.time()
        img_emb = encode_query_image_only(image_path)
        log(f"⏱️ Encode image query xong: {time.time() - t0:.2f}s")
        t0 = time.time()
        text_emb, caption = encode_query_caption(image_path)
        if caption.strip() == "":
            log("⚠️ Caption sinh ra rỗng hoặc lỗi, fallback sang image embedding thuần")
            query_emb = img_emb
        else:
            log(f"📝 Caption sinh bởi BLIP-2: {caption}")
            query_emb = combine_embeddings(img_emb, text_emb, alpha=0.7)
            query_emb = normalize_emb(query_emb)
            log("✅ Kết hợp image + text embedding hoàn tất")
        log(f"⏱️ Encode caption query xong: {time.time() - t0:.2f}s")

    else:
        log("⚠️ Loại query không hợp lệ.")
        return

    t0 = time.time()
    results = search_topk(index, metas, query_emb, TOP_K)
    log(f"⏱️ Search xong: {time.time() - t0:.2f}s\n")

    if not results or len(results[0]) == 0:
        log("🔍 Không tìm thấy kết quả.")
    else:
        log(f"🎬 Top-{TOP_K} frames tương đồng:")
        for r in results[0]:
            log(f" {r['rank']:>2}. Score={r['score']:.4f} | Video={r['video_id']} | Frame={r['frame_id']}")
            log(f"     Path: {r['path']} | Timestamp: {r['timestamp']}")
        show_best_frame(results)

    log("\n✅ Done.")

# ============================ 
# Run
# ============================
if __name__ == "__main__":
    main()
