from fastapi import FastAPI,HTTPException
import config # noqa: F401
from middleware import CustomResponseMiddleware
from exception_handle import http_exception_handler, generic_exception_handler

# Import các controller (router)
from controllers.video_controller import router as video_router
from controllers.auth_controller import router as user_router
from controllers.category_controller import router as category_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="My Video App API",
        version="1.0.0"
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
