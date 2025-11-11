from database.connection import db #database
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List

USERS_COLLECTION = "users"
class LikeService:
    def __init__(self, db: AsyncIOMotorClient):
        self.users_collection = db[USERS_COLLECTION]

    async def add_liked_video(self, user_id: int, video_id: int) -> int:
        """Thêm video_id vào liked_videos bằng $addToSet."""
        result = await self.users_collection.update_one(
            {"_id": user_id},
            {
                # $addToSet: Thêm nếu chưa tồn tại
                "$addToSet": {"liked_videos": video_id}
            }
        )
        return result.modified_count
    async def remove_liked_video(self, user_id: int, video_id: int) -> int:
            """Xóa video_id khỏi liked_videos bằng $pull."""
            result = await self.users_collection.update_one(
                {"_id": user_id},
                {
                    # $pull: Xóa tất cả các lần xuất hiện của giá trị khỏi mảng
                    "$pull": {"liked_videos": video_id}
                }
            )
            return result.modified_count