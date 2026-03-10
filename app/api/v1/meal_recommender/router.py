from fastapi import APIRouter, HTTPException

from app.api.v1.meal_recommender.schemas import (
    CaloriesCalculationRequest,
    CaloriesCalculationResponse,
    DailyMealRecommendation,
    MealRecommendRequest,
)
from app.api.v1.meal_recommender.service import calculate_calories, recommend_meal


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


@router.post("/calculate-calories", response_model=CaloriesCalculationResponse)
async def calculate_calories_endpoint(payload: CaloriesCalculationRequest):
    """
    유저가 작성한 식단(음식 이름 + 섭취량 g)을 받아 항목별 추정 칼로리와 영양소를 계산합니다.
    """
    try:
        result = await calculate_calories(payload)
        return result
    except Exception as e:
        print(f"Error during calories calculation: {e}")
        raise HTTPException(
            status_code=500,
            detail="칼로리 계산 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

