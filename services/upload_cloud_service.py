import cloudinary.uploader

async def upload_file_to_cloud(file, resource_type="auto") -> str:
    """
    Upload file lên Cloudinary và trả về secure_url.
    """
    result = cloudinary.uploader.upload_large(
        file,
        resource_type=resource_type
    )
    return result.get("secure_url")
