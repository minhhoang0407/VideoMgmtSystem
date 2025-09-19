from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import json
from response_formatter import success_response

class CustomResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Bỏ qua Swagger & OpenAPI docs
        if request.url.path in ["/openapi.json", "/docs", "/redoc"]:
            return await call_next(request)

        response = await call_next(request)

        if (
            response.status_code == 200
            and response.headers.get("content-type") == "application/json"
        ):
            body = b"".join([chunk async for chunk in response.body_iterator])

            try:
                data = json.loads(body) if body else None
            except Exception:
                data = body.decode() if body else None
                
            # ✅ chỉ giữ lại Set-Cookie
            headers = {}
            if "set-cookie" in response.headers:
                headers["set-cookie"] = response.headers["set-cookie"]

            return JSONResponse(
                content=success_response(message="Success",data=data),
                status_code=200,
                headers=headers
            )

        return response
