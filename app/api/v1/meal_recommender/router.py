from fastapi import APIRouter, HTTPException

from app.api.v1.meal_recommender.schemas import (
    DailyMealRecommendation,
    MealRecommendRequest,
)
from app.api.v1.meal_recommender.service import recommend_meal


router = APIRouter()


@router.post("/recommend", response_model=DailyMealRecommendation)
async def recommend_meal_endpoint(payload: MealRecommendRequest):
    """
    목적/알러지에 맞는 아침·점심·저녁 식단을 한 번에 추천합니다.
    맘에 들지 않으면 disliked_items에 제외할 메뉴/재료를 넣고 다시 호출하면 전부 새로 추천합니다.
    """
    try:
        result = await recommend_meal(payload)
        return result
    except Exception as e:
        # 에러 로그 출력 (디버깅용)
        print(f"Error during meal recommendation: {e}")
        raise HTTPException(
            status_code=500,
            detail="식단 추천 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

