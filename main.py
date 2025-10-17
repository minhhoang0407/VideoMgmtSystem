from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config # noqa: F401
from middleware import CustomResponseMiddleware
from exception_handle import http_exception_handler, generic_exception_handler

# Import các controller (router)
from controllers.video_controller import router as video_router
from controllers.auth_controller import router as user_router
from controllers.category_controller import router as category_router

import logging
logging.getLogger("uvicorn.access").addFilter(
    lambda record: "OPTIONS" not in record.getMessage()
)

def create_app() -> FastAPI:
    app = FastAPI(
        title="My Video App API",
        version="1.0.0"
    )

    # ✅ Add CORS middleware
    origins = [
        "http://localhost:5174",   # FE chạy Vite
        "http://127.0.0.1:5174",
        "http://localhost:5175",   # thêm cho chắc
        "http://127.0.0.1:5175",
        
              # backup cho localhost
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,      # chỉ cho phép FE gọi
        allow_credentials=True,
        allow_methods=["*"],        # GET, POST, PUT, DELETE...
        allow_headers=["*"],        # cho tất cả headers
    )
    # ✅ Add middleware 
    app.add_middleware(CustomResponseMiddleware)

    # ✅ Add exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    # Đăng ký routers
    app.include_router(user_router)
    app.include_router(video_router)
    app.include_router(category_router)
    
    return app

app = create_app()
