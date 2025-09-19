from fastapi.responses import JSONResponse

def success_response(message: str = "Success", data: dict | list | None = None):
    return {
        "success": True,
        "message": message,
        "extensions": {
            "code": "SUCCESS",
            "status": 200,
            "data": data,
        }
    }

def error_response(message: str, code: str = "ERROR", status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "extensions": {
                "code": code,
                "status": status_code
            }
        }
    )