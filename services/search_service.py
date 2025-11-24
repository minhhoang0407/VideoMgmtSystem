import numpy as np
import torch
import open_clip
import faiss
from PIL import Image
from io import BytesIO
from googletrans import Translator
import asyncio
import json

# Import BLIP-2
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Thêm sys.path để import database
import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from database.connection import db

# Đường dẫn tới các tài sản
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
EMBEDDINGS_NPY = os.path.join(BASE_DIR, "cached_frame_embs_RESYNCED.npy")
IDS_JSON = os.path.join(BASE_DIR, "frame_ids_RESYNCED.json")
FRAMES_COLLECTION = "frames"

class SearchService:
    def __init__(self):
        self.clip_model = None
        self.clip_preprocess = None

        # CẬP NHẬT: Thêm BLIP-2 và trọng số alpha
        self.blip_processor = None 
        self.blip_model = None
        self.alpha = 0.7 # Trọng số kết hợp (giống pipeline_retrieval.py)

        self.translator = None
        self.faiss_index = None
        self.frame_ids = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load(self):
        """
        Hàm khởi động chính.
        - Load Model AI (chỉ 1 lần).
        - Load Dữ liệu Index (gọi hàm con).
        """
        print("INFO:     Loading Search Service assets...")
        
        # 1. Tải Models (Chỉ tải nếu chưa có)
        if self.clip_model is None:
            print("INFO:     Loading CLIP model...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", quick_gelu=True
            )
            self.clip_model = self.clip_model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

            print("INFO:     Loading BLIP-2 model (có thể mất vài phút)...")
            blip_model_id = "Salesforce/blip2-opt-2.7b"
            self.blip_processor = Blip2Processor.from_pretrained(blip_model_id, use_fast=True)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(blip_model_id).to(self.device)
            
            self.translator = Translator()
            print("INFO:     AI Models Loaded Successfully.")

        # 2. Tải dữ liệu tìm kiếm (Index)
        await self.reload_indices()
        print(f"INFO:     Search Service ready.")

    async def reload_indices(self):
        """
        Hàm này chuyên dùng để Refresh dữ liệu tìm kiếm.
        Có thể gọi lại nhiều lần mà không ảnh hưởng đến Model AI.
        """
        print("INFO:     Reloading Search Indices (NPY & JSON)...")

        # A. Kiểm tra file tồn tại
        if not os.path.exists(EMBEDDINGS_NPY) or not os.path.exists(IDS_JSON):
            print("WARNING:  Chưa có dữ liệu embedding hoặc mapping. Hệ thống tìm kiếm tạm thời rỗng.")
            self.faiss_index = None
            self.frame_ids = []
            return

        try:
            # B. Load Embeddings (.npy)
            all_embeddings = np.load(EMBEDDINGS_NPY).astype('float32')
            if all_embeddings.shape[0] == 0:
                print("WARNING:  File embedding rỗng.")
                self.faiss_index = None
                self.frame_ids = []
                return

            # C. Build FAISS Index
            embedding_dim = all_embeddings.shape[1]
            new_index = faiss.IndexFlatIP(embedding_dim)
            faiss.normalize_L2(all_embeddings)
            new_index.add(all_embeddings)
            
            # Gán vào biến class (Thread-safe đơn giản bằng việc gán pointer)
            self.faiss_index = new_index

            # D. Load ID Mapping (.json)
            with open(IDS_JSON, 'r') as f:
                self.frame_ids = json.load(f)

            # E. Sanity Check
            num_vectors = self.faiss_index.ntotal
            num_ids = len(self.frame_ids)

            if num_vectors != num_ids:
                print(f"❌ CRITICAL ERROR: Dữ liệu bị lệch! Vectors: {num_vectors} != IDs: {num_ids}")
            else:
                print(f"✅ Index Reloaded: {num_ids} items loaded.")

        except Exception as e:
            print(f"❌ Error reloading indices: {e}")

#==================ENCODING=======================
    #Hàm helper chuẩn hóa L2
    def _normalize_emb(self, emb):
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / (norm + 1e-10) # Thêm 1e-10 để tránh chia cho 0

    #Hàm helper kết hợp embedding
    def _combine_embeddings(self, img_emb, text_emb, alpha):
        if img_emb.shape != text_emb.shape:
            print("WARNING: Kích thước embedding khác nhau, sử dụng image embedding thuần")
            return img_emb
        combined = alpha * img_emb + (1 - alpha) * text_emb
        return combined

    #Hàm helper encode ảnh (CLIP thuần)
    def _encode_image_clip(self, image: Image.Image):
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image_tensor)
            # Không chuẩn hóa ở đây vội, chuẩn hóa sau khi kết hợp
        return image_embedding.cpu().numpy()

    #Hàm helper sinh caption (BLIP-2) và encode (CLIP)
    def _encode_caption_blip(self, image: Image.Image):
        try:
            # 1. Sinh caption bằng BLIP-2
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            if not caption.strip():
                return np.empty((1, 768), dtype=np.float32), ""

            # 2. Encode caption bằng CLIP
            tokens = self.tokenizer([caption]).to(self.device)
            with torch.no_grad():
                text_embedding = self.clip_model.encode_text(tokens)
            return text_embedding.cpu().numpy(), caption
        
        except Exception as e:
            print(f"ERROR: Lỗi khi sinh caption BLIP-2: {e}")
            return np.empty((1, 768), dtype=np.float32), ""

    #Hàm _search
    def _search(self, query_embedding, top_k=10):
        """
        Hàm tìm kiếm cốt lõi bằng FAISS.
        """
        #Phải chuẩn hóa L2 vector query TRƯỚC KHI tìm kiếm
        faiss.normalize_L2(query_embedding)

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
            
        D, I = self.faiss_index.search(query_embedding, max(50, top_k * 5)) 
        
        potential_indices = I[0]
        selected_frame_ids = []
        selected_video_ids = set()
        
        for index in potential_indices:
            if index >= len(self.frame_ids):
                continue
            frame_id = self.frame_ids[index]
            try:
                video_id = "_".join(frame_id.split('_')[:-1]) 
            except:
                video_id = frame_id
                
            if video_id not in selected_video_ids:
                selected_frame_ids.append(frame_id)
                selected_video_ids.add(video_id)
                if len(selected_frame_ids) >= top_k:
                    break
        
        return selected_frame_ids

    async def search_by_text(self, text: str, top_k=10):
        translated_text = text
        try:
            # Thêm timeout
            translation = await self.translator.translate(text, dest='en')
            translated_text = translation.text
        except Exception as e:
            print(f"⚠️ Translator failed: {e}. Using original text.")
            # Nếu dịch lỗi, dùng luôn text gốc để tìm kiếm thay vì crash app
            translated_text = text
        
        # Mã hóa văn bản
        text_tokens = self.tokenizer([translated_text]).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_tokens)
        
        query_vector = text_embedding.cpu().numpy().astype('float32')
        # Phải dùng to_thread vì _search là sync
        loop = asyncio.get_event_loop()
        found_ids = await loop.run_in_executor(None, self._search, query_vector, top_k)
        return found_ids

    #
    async def search_by_image(self, image_bytes: bytes, top_k=10):
        """
        Hàm search_by_image được cập nhật để chạy bất đồng bộ,
        không làm block server.
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Bước 1: Encode ảnh (CLIP) - Chạy trên thread riêng
        img_emb = await asyncio.to_thread(self._encode_image_clip, image)

        # # Bước 2: Sinh caption (BLIP-2) - Chạy trên thread riêng
        text_emb, caption = await asyncio.to_thread(self._encode_caption_blip, image)

        print(f"INFO: BLIP-2 Caption: {caption}")

        # # Bước 3: Kết hợp (Phần này nhanh, chạy ngay)
        if not caption.strip():
            print("WARNING: Caption rỗng, fallback về image embedding thuần")
            query_vector = self._normalize_emb(img_emb)
        else:
            print("INFO: Kết hợp Image + Text embedding")
            combined_emb = self._combine_embeddings(img_emb, text_emb, self.alpha)
            query_vector = self._normalize_emb(combined_emb)
        query_vector = query_vector.astype('float32')

        #Dùng khi sử dụng image embedding thuần túy CLIP
        # print("INFO: Sử dụng image embedding thuần túy (BLIP-2 đã tắt)")
        # query_vector = self._normalize_emb(img_emb)
        # query_vector = query_vector.astype('float32')

        # Bước 4: Tìm kiếm - Chạy trên thread riêng
        found_ids = await asyncio.to_thread(self._search, query_vector, top_k)
        return found_ids

# Tạo một instance duy nhất của service để sử dụng trong toàn bộ ứng dụng
search_service = SearchService()