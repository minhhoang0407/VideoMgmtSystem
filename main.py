import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config  # noqa: F401
from middleware import CustomResponseMiddleware
from exception_handle import http_exception_handler, generic_exception_handler

# Import các controller (router)
from controllers.video_controller import router as video_router
from controllers.auth_controller import router as user_router
from controllers.category_controller import router as category_router
from controllers.frame_controller import router as frame_router

# 1. IMPORT HÀM VÒNG LẶP CỦA WORKER
from model_trainned.cut_frames_worker import main_worker_loop
from services.search_service import search_service

import logging
logging.getLogger("uvicorn.access").addFilter(
    lambda record: "OPTIONS" not in record.getMessage()
)

# 2. TẠO LIFESPAN MANAGER ĐỂ QUẢN LÝ WORKER
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng.
    - Khi server khởi động: Chạy worker trong một tác vụ nền.
    - Khi server tắt: Dừng tác vụ nền của worker.
    """
    print("INFO:     🚀 Server is starting up, initializing background worker...")
    
    # Tạo một tác vụ nền cho worker, nó sẽ chạy song song với server
    worker_task = asyncio.create_task(main_worker_loop())

    # Tải các model và index của cỗ máy tìm kiếm
    # Chạy trong threadpool để không block event loop chính khi tải file
    await search_service.load()
    # Server đã sẵn sàng và bắt đầu nhận request
    yield
    
    # Logic dọn dẹp khi server tắt (ví dụ: nhấn Ctrl+C)
    print("INFO:     🛑 Server is shutting down, stopping worker...")
    worker_task.cancel()
    try:
        # Chờ tác vụ worker thực sự bị hủy
        await worker_task
    except asyncio.CancelledError:
        print("INFO:     ✅ Background worker stopped successfully.")


def create_app() -> FastAPI:
    # 3. KHỞI TẠO APP VỚI LIFESPAN MANAGER
    app = FastAPI(
        title="My Video App API",
        version="1.0.0",
        lifespan=lifespan
    )

    # ✅ Add CORS middleware (Giữ nguyên)
    origins = [
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5180",# thêm cho chắc
        "http://127.0.0.1:5180",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # ✅ Add middleware (Giữ nguyên)
    app.add_middleware(CustomResponseMiddleware)

    # ✅ Add exception handlers (Giữ nguyên)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    # Đăng ký routers (Giữ nguyên)
    app.include_router(user_router)
    app.include_router(video_router)
    app.include_router(frame_router)
    app.include_router(category_router)
    
    return app

app = create_app()