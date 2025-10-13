from fastapi.concurrency import run_in_threadpool
import cloudinary.uploader

async def upload_file_to_cloud(file, uploader_name: str, resource_type="auto") -> str:
    """
    Upload file lên Cloudinary (chạy trong threadpool để tránh block asyncio)
    """
    def _upload():
        return cloudinary.uploader.upload(
            file,
            resource_type=resource_type,
            folder=f"videos/{uploader_name}/"
        )

    result = await run_in_threadpool(_upload)
    #print("Cloudinary result:", result)
    return result.get("secure_url")

