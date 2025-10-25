import numpy as np
import torch
import open_clip
import faiss
from PIL import Image
from io import BytesIO
from googletrans import Translator

# Thêm sys.path để import database
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import db

# Đường dẫn tới các tài sản
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
EMBEDDINGS_NPY = os.path.join(BASE_DIR, "cached_frame_embs_RESYNCED.npy")
FRAMES_COLLECTION = "frames"

class SearchService:
    def __init__(self):
        self.clip_model = None
        self.clip_preprocess = None
        self.blip_model = None # Tạm thời chưa tích hợp BLIP-2 để đơn giản hóa
        self.translator = None
        self.faiss_index = None
        self.frame_ids = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load(self):
        """
        Tải tất cả các model và dữ liệu cần thiết.
        Hàm này chỉ được gọi một lần khi server khởi động.
        """
        print("INFO:     Loading Search Service assets...")
        
        # 1. Tải CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", quick_gelu=True
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

        # 2. Tải Google Translator
        self.translator = Translator()

        # 3. Tải embeddings và xây dựng FAISS index
        if not os.path.exists(EMBEDDINGS_NPY):
            raise FileNotFoundError(f"File embedding không tồn tại: {EMBEDDINGS_NPY}")

        all_embeddings = np.load(EMBEDDINGS_NPY).astype('float32')
        if all_embeddings.shape[0] == 0:
            print("WARNING:  File embedding rỗng, tìm kiếm sẽ không hoạt động.")
            return

        embedding_dim = all_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(all_embeddings)

        # 4. Tải danh sách frame_id từ MongoDB để map kết quả
        # Đảm bảo thứ tự của frame_id khớp với thứ tự của embedding
        frame_cursor = db[FRAMES_COLLECTION].find({}, {"frame_id": 1}).sort([
            ("video_id", 1), 
            ("frame_id", 1)
        ])
                
        self.frame_ids = [doc["frame_id"] async for doc in frame_cursor]

        print(f"INFO: Search Service loaded successfully. Indexed {self.faiss_index.ntotal} frames.")

    def _search(self, query_embedding, top_k=10):
        """Hàm tìm kiếm cốt lõi bằng FAISS."""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
            
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Lấy ra các frame_id tương ứng từ chỉ số
        results = [self.frame_ids[i] for i in indices[0] if i < len(self.frame_ids)]
        return results

    async def search_by_text(self, text: str, top_k=10):
        # Dịch sang tiếng Anh
        translation = await self.translator.translate(text, dest='en')
        translated_text = translation.text
        
        # Mã hóa văn bản
        text_tokens = self.tokenizer([translated_text]).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        
        query_vector = text_embedding.cpu().numpy().astype('float32')
        return self._search(query_vector, top_k)

    def search_by_image(self, image_bytes: bytes, top_k=10):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image_tensor)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        
        query_vector = image_embedding.cpu().numpy().astype('float32')
        return self._search(query_vector, top_k)

# Tạo một instance duy nhất của service để sử dụng trong toàn bộ ứng dụng
search_service = SearchService()