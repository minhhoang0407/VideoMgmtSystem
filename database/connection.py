from motor.motor_asyncio import AsyncIOMotorClient
from config import mongodb_uri
import certifi

#connect database
client = AsyncIOMotorClient(mongodb_uri)
db = client.get_database("video_app")


# #connect database
# client = AsyncIOMotorClient("mongodb://localhost:27017")
# db = client.get_database("video_platform")