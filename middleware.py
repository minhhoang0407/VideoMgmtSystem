# from fastapi import Request
# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.responses import JSONResponse
# import json
# from response_formatter import success_response

# class CustomResponseMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         # Bỏ qua Swagger & OpenAPI docs
#         if request.url.path in ["/openapi.json", "/docs", "/redoc"]:
#             return await call_next(request)

#         response = await call_next(request)

#         if (
#             response.status_code == 200
#             and response.headers.get("content-type") == "application/json"
#         ):
#             body = b"".join([chunk async for chunk in response.body_iterator])

#             try:
#                 data = json.loads(body) if body else None
#             except Exception:
#                 data = body.decode() if body else None
                
#             # ✅ chỉ giữ lại Set-Cookie
#             headers = {}
#             if "set-cookie" in response.headers:
#                 headers["set-cookie"] = response.headers["set-cookie"]

#             return JSONResponse(
#                 content=success_response(message="Success",data=data),
#                 status_code=200,
#                 headers=headers
#             )

#         return response


#-------------Trang (fixed)------------
# from fastapi import Request
# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.responses import JSONResponse
# import json
# from starlette.responses import FileResponse
# from response_formatter import success_response

# class CustomResponseMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         if request.url.path in ["/openapi.json", "/docs", "/redoc"]:
#             return await call_next(request)

#         response = await call_next(request)

#         if isinstance(response, FileResponse):
#             return response

#         # Đọc body gốc
#         body = b"".join([chunk async for chunk in response.body_iterator])
#         async def new_body():
#             yield body
#         response.body_iterator = new_body()

#         try:
#             data = json.loads(body) if body else None
#         except Exception:
#             data = body.decode() if body else None

#         # ✅ Bỏ qua nếu đã có "success"
#         if isinstance(data, dict) and "success" in data:
#             return JSONResponse(
#                 content=data,
#                 status_code=response.status_code,
#                 headers=response.headers
#             )

#         # ✅ Nếu chưa có success thì format lại
#         if response.status_code == 200:
#             formatted = success_response(message="Success", data=data)
#             return JSONResponse(
#                 content=formatted,
#                 status_code=200,
#                 headers=response.headers
#             )

#         # ✅ Format lỗi
#         return JSONResponse(
#             content={
#                 "success": False,
#                 "message": "Error",
#                 "extensions": {
#                     "code": "ERROR",
#                     "status": response.status_code,
#                     "data": data,
#                 },
#             },
#             status_code=response.status_code,
#             headers=response.headers
#         )

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
import json
from response_formatter import success_response

class CustomResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ["/openapi.json", "/docs", "/redoc"]:
            return await call_next(request)

        response = await call_next(request)

        # ⚠ Bỏ qua mọi response không phải JSON
        if not isinstance(response, JSONResponse):
            return response

        # Đọc body gốc
        body = b"".join([chunk async for chunk in response.body_iterator])
        async def new_body():
            yield body
        response.body_iterator = new_body()

        try:
            data = json.loads(body) if body else None
        except Exception:
            data = body.decode() if body else None

        # Nếu đã có success → trả thẳng
        if isinstance(data, dict) and "success" in data:
            return JSONResponse(
                content=data,
                status_code=response.status_code,
                headers=response.headers
            )

        # Nếu 200 → wrap success_response
        if response.status_code == 200:
            formatted = success_response(message="Success", data=data)
            return JSONResponse(
                content=formatted,
                status_code=200,
                headers=response.headers
            )

        # Format lỗi
        return JSONResponse(
            content={
                "success": False,
                "message": "Error",
                "extensions": {
                    "code": "ERROR",
                    "status": response.status_code,
                    "data": data,
                },
            },
            status_code=response.status_code,
            headers=response.headers
        )
