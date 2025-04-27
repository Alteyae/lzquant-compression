"""
API v1 - PLZ compression format endpoints.
"""
from fastapi import APIRouter
from app.api.v1.plz import router as plz_router

router = APIRouter()
router.include_router(plz_router)