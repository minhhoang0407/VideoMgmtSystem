from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from response_formatter import error_response

# ✅ Handler cho HTTPException
async def http_exception_handler(request: Request, exc: HTTPException):
    return error_response(
        message=exc.detail,
        code=exc.__class__.__name__,
        status_code=exc.status_code
    )

# ✅ Handler cho Exception thường
async def generic_exception_handler(request: Request, exc: Exception):
    return error_response(
        message=str(exc) or "Unexpected error",
        code="INTERNAL_ERROR",
        status_code=500
    )
