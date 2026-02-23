from fastapi import APIRouter
from app.api.v1.diet_analyzer.router import router as diet_analyzer_router

api_router = APIRouter()
api_router.include_router(diet_analyzer_router, prefix="/dietanalyzer", tags=["Dietanalyzer"])


@api_router.get("/health", tags=["Health"])
async def health_check():
    """서버 정상 작동 여부 확인용 테스트 API."""
    return {"status": "ok", "message": "LB Python API is running"}