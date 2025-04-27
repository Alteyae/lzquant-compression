"""
API v2 - Zstandard compression format endpoints.
"""
from fastapi import APIRouter
from app.api.v2.zst import router as zst_router

router = APIRouter()
router.include_router(zst_router)