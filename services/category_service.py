from bson import ObjectId
from database.connection import db
from models.category import CategoryRequest, CategoryResponse,CategoryUpdateRequest
from fastapi import HTTPException

collection = db["categories"]

# ================== Helper ==================
def category_helper(cat) -> CategoryResponse:
    return CategoryResponse(
        _id=cat["_id"],  # ép sang string để trả ra API
        name=cat["name"],
        description=cat.get("description")
    )


async def get_next_id_from_max():
    """Sinh ID mới dựa trên max _id hiện có"""
    last = await collection.find_one(sort=[("_id", -1)])
    return (last["_id"] + 1) if last and "_id" in last else 1


# 🔹 Create
async def create_category(req: CategoryRequest) -> CategoryResponse:
    new_cat = req.model_dump()
    new_cat["_id"] = await get_next_id_from_max()  # int ID
    await collection.insert_one(new_cat)
    return category_helper(new_cat)

# 🔹 Read all
async def get_all_categories():
    categories = []
    async for cat in collection.find():
        categories.append(category_helper(cat))
    return categories

# 🔹 Read by id
async def get_category_by_id(category_id: int):
    cat = await collection.find_one({"_id": category_id})
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    return category_helper(cat)

# 🔹 Update
async def update_category(category_id: int, req: CategoryUpdateRequest):
    update_data = req.model_dump(exclude_unset=True)
    result = await collection.update_one(
        {"_id": category_id},
        {"$set": update_data}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Category not found or no changes made")
    updated = await collection.find_one({"_id": category_id})
    return category_helper(updated)

# 🔹 Delete
async def delete_category(category_id: int):
    result = await collection.delete_one({"_id": category_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"message": "Category deleted successfully"}
