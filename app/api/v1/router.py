from fastapi import APIRouter

from app.api.v1.diet_analyzer.router import router as diet_analyzer_router
from app.api.v1.meal_recommender.router import router as meal_recommender_router
from app.api.v1.chatbot.router import router as chatbot_router

api_router = APIRouter()
api_router.include_router(diet_analyzer_router, prefix="/dietanalyzer", tags=["Dietanalyzer"])
api_router.include_router(meal_recommender_router, prefix="/meal", tags=["MealRecommender"])
api_router.include_router(chatbot_router)

@api_router.get("/health", tags=["Health"])
async def health_check():
    """서버 정상 작동 여부 확인용 테스트 API."""
    return {"status": "ok", "message": "LB Python API is running"}