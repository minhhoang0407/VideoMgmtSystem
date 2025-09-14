from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
import json
from response_formatter import success_response

class CustomResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # ✅ Bỏ qua Swagger & OpenAPI docs
        if request.url.path in ["/openapi.json", "/docs", "/redoc"]:
            return await call_next(request)

        response = await call_next(request)

        # ✅ Chỉ wrap JSON 200
        if (
            response.status_code == 200
            and response.headers.get("content-type") == "application/json"
        ):
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            if not body or body == b"null":
                data = None
            else:
                try:
                    data = json.loads(body)
                except Exception:
                    data = body.decode()

            wrapped = success_response(
                message="Success",
                data=data
            )
            return JSONResponse(content=wrapped, status_code=200)

        # 🔑 Trả lại response gốc cho các trường hợp khác (error, file, ...)
        return response
