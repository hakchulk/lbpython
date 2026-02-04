from fastapi import APIRouter
from app.api.v1.diet_analyzer.router import router as diet_analyzer_router

api_router = APIRouter()
api_router.include_router(diet_analyzer_router, prefix="/dietanalyzer", tags=["Dietanalyzer"])