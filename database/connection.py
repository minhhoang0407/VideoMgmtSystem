from motor.motor_asyncio import AsyncIOMotorClient
from config import mongodb_uri

#connect database
client = AsyncIOMotorClient(mongodb_uri)
db = client.get_database("video_app")