from bson import ObjectId
from database.connection import db
from models.category import CategoryRequest, CategoryResponse
from fastapi import HTTPException

collection = db["categories"]

def category_helper(cat) -> CategoryResponse:
    return CategoryResponse(
        id=str(cat["_id"]),
        name=cat["name"],
        description=cat.get("description")
    )

# 🔹 Create
async def create_category(req: CategoryRequest) -> CategoryResponse:
    new_cat = req.model_dump()
    result = await collection.insert_one(new_cat)
    created = await collection.find_one({"_id": result.inserted_id})
    return category_helper(created)

# 🔹 Read all
async def get_all_categories():
    categories = []
    async for cat in collection.find():
        categories.append(category_helper(cat))
    return categories

# 🔹 Read by id
async def get_category_by_id(category_id: str):
    cat = await collection.find_one({"_id": ObjectId(category_id)})
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    return category_helper(cat)

# 🔹 Update
async def update_category(category_id: str, req: CategoryRequest):
    update_data = req.model_dump()
    result = await collection.update_one(
        {"_id": ObjectId(category_id)},
        {"$set": update_data}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Category not found or no changes made")
    updated = await collection.find_one({"_id": ObjectId(category_id)})
    return category_helper(updated)

# 🔹 Delete
async def delete_category(category_id: str):
    result = await collection.delete_one({"_id": ObjectId(category_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"message": "Category deleted successfully"}
