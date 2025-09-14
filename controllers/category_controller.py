from fastapi import APIRouter, Depends
from typing import List
from models.category import CategoryRequest, CategoryResponse
from services.category_service import (
    create_category,
    get_all_categories,
    get_category_by_id,
    update_category,
    delete_category,
)
from services.auth_service import get_current_user  # ✅ để xác thực

router = APIRouter(prefix="/categories", tags=["Categories"])

# 🔹 Public GET
@router.get("/", response_model=List[CategoryResponse])
async def list_categories():
    return await get_all_categories()

@router.get("/{category_id}", response_model=CategoryResponse)
async def get_category(category_id: str):
    return await get_category_by_id(category_id)

# 🔹 Auth required
@router.post("/", response_model=CategoryResponse)
async def add_category(req: CategoryRequest, current_user: dict = Depends(get_current_user)):
    return await create_category(req)

@router.put("/{category_id}", response_model=CategoryResponse)
async def edit_category(category_id: str, req: CategoryRequest, current_user: dict = Depends(get_current_user)):
    return await update_category(category_id, req)

@router.delete("/{category_id}")
async def remove_category(category_id: str, current_user: dict = Depends(get_current_user)):
    return await delete_category(category_id)
