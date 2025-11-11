import numpy as np
import torch
import open_clip
import faiss
from PIL import Image
from io import BytesIO
from googletrans import Translator
import asyncio

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
        print(f"INFO: Indexed {self.faiss_index.ntotal} frames in FAISS.")
        print(f"INFO: Loaded {len(self.frame_ids)} frame_ids from database.")

    def _search(self, query_embedding, top_k=10):
        """
        Hàm tìm kiếm cốt lõi bằng FAISS, được chỉnh sửa để chỉ trả về
        frame từ các VIDEO KHÁC NHAU cho đến khi đạt top_k.
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
            
        # 1. Tìm kiếm tất cả các kết quả tiềm năng (nên lấy nhiều hơn top_k ban đầu)
        # Ví dụ: Tìm top 100 kết quả tiềm năng
        D, I = self.faiss_index.search(query_embedding, max(50, top_k * 5)) 
        
        # Lấy danh sách indices từ kết quả tìm kiếm
        potential_indices = I[0]
        
        selected_frame_ids = []
        selected_video_ids = set()
        
        # 2. Lặp qua các chỉ mục tiềm năng để chọn lọc (Diversification)
        for index in potential_indices:
            # Kiểm tra xem chỉ mục có hợp lệ không
            if index >= len(self.frame_ids):
                continue

            frame_id = self.frame_ids[index]
            
            # Giả định frame_id có dạng: {video_id}_{frame_index}
            # Cần tách video_id
            try:
                # Tìm dấu "_" cuối cùng để tách video_id ra khỏi frame index
                video_id = "_".join(frame_id.split('_')[:-1]) 
            except:
                # Trường hợp frame_id không theo định dạng mong muốn, sử dụng toàn bộ frame_id
                video_id = frame_id
                
            # Logic quan trọng: Kiểm tra xem video này đã được chọn chưa
            if video_id not in selected_video_ids:
                # Nếu chưa, thêm frame_id này vào kết quả và đánh dấu video đã chọn
                selected_frame_ids.append(frame_id)
                selected_video_ids.add(video_id)
                
                # Nếu đã đủ số lượng top_k, dừng lại
                if len(selected_frame_ids) >= top_k:
                    break
        
        return selected_frame_ids

    async def search_by_text(self, text: str, top_k=10):
        # Dịch sang tiếng Anh
        translation =await self.translator.translate(text, dest='en')
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