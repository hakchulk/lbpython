from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.api.v1.diet_analyzer.schemas import NutrientInfo


class DietGoal(str, Enum):
    WEIGHT_LOSS = "체중감량"
    HEALTH_MAINTENANCE = "건강유지"
    MUSCLE_GAIN = "근육량증가"
    BLOOD_SUGAR_CONTROL = "혈당관리"
    CHOLESTEROL_CONTROL = "콜레스테롤"


class AllergyType(str, Enum):
    MILK = "우유"
    EGG = "달걀"
    FISH = "생선"
    CRUSTACEAN = "갑각류"
    TREE_NUTS = "건과류"
    PEANUT = "땅콩"
    WHEAT = "밀"


class MealType(str, Enum):
    BREAKFAST = "아침"
    LUNCH = "점심"
    DINNER = "저녁"


class UserProfile(BaseModel):
    """스프링 Member 엔티티 기반 개인 정보."""

    member_id: int = Field(..., description="회원 ID")
    name: str = Field(..., description="이름")
    gender: str = Field(..., description="성별 (예: M/F)")
    birthday: date = Field(..., description="생년월일")
    height: float = Field(..., description="키(cm)")
    weight: float = Field(..., description="현재 체중(kg)")
    goal: str = Field(..., description="회원이 설정한 목표 설명 (예: 중 감량, 건강 유지 등)")
    goal_weight: float = Field(..., description="목표 체중(kg)")
    target_date: int = Field(..., description="목표까지 남은 일수")
    allergies: List[str] = Field(
        default_factory=list,
        description="회원이 입력한 피해야 할 음식 리스트 (원문 문자열 기준)",
    )
    special_notes: str = Field(
        default="",
        description="특이사항 (예: 고기 안먹음 등 식습관/제약 사항)",
    )
    daily_calories: int = Field(..., description="하루 섭취 목표 칼로리 (kcal)")


class MealRecommendRequest(BaseModel):
    goal: DietGoal = Field(..., description="식단 추천 목적 (체중감량, 건강유지, 근육량증가, 혈당관리, 콜레스테롤)")
    allergies: List[AllergyType] = Field(
        default_factory=list,
        description="피해야 할 알러지 식품 목록 (우유, 달걀, 생선, 갑각류, 건과류, 땅콩, 밀)",
    )
    disliked_items: List[str] = Field(
        default_factory=list,
        description="맘에 들지 않아 전부 새로 받을 때, 제외할 메뉴/재료 (이전 추천에서 싫었던 것)",
    )
    profile: Optional[UserProfile] = Field(
        default=None,
        description="회원 신체/목표/식습관 정보를 담은 프로필 (있으면 개인 맞춤 강하게 반영)",
    )


class IngredientPortion(BaseModel):
    """재료별 1인분 권장 섭취량(g)."""

    name: str = Field(..., description="재료명 (예: 닭가슴살, 현미)")
    grams: int = Field(..., description="1인분 기준 권장 그람수 (g)")


class RecommendedMenuItem(BaseModel):
    name: str = Field(..., description="추천 메뉴 이름")
    description: str = Field(..., description="메뉴에 대한 간단한 설명")
    ingredients: List[IngredientPortion] = Field(
        ...,
        description="주요 재료 목록 (재료명 + 1인분 기준 권장 그람수)",
    )
    calories: int = Field(..., description="예상 칼로리 (kcal)")
    nutrients: NutrientInfo = Field(..., description="해당 메뉴의 주요 영양 성분")


class DailyMealRecommendation(BaseModel):
    """아침·점심·저녁 한 번에 추천 결과."""

    goal: DietGoal = Field(..., description="사용자가 설정한 식단 목적")
    breakfast: RecommendedMenuItem = Field(..., description="아침 메뉴")
    lunch: RecommendedMenuItem = Field(..., description="점심 메뉴")
    dinner: RecommendedMenuItem = Field(..., description="저녁 메뉴")
    comment: str = Field(..., description="오늘 하루 식단을 이렇게 추천한 이유와 코멘트")
    cautions: List[str] = Field(
        default_factory=list,
        description="알러지, 질환(혈당/콜레스테롤 등)에 따른 주의사항 및 보완 팁",
    )


# --- 음식 이름 + 양 기준 칼로리 계산 ---


class FoodEntryForCalories(BaseModel):
    """칼로리 계산 요청용: 음식 이름 + 섭취량(g)."""

    name: str = Field(..., description="음식 이름 (예: 닭가슴살, 현미밥)")
    amount_grams: int = Field(..., ge=1, description="섭취량 (g)")


class CaloriesCalculationRequest(BaseModel):
    """유저가 작성한 식단(이름 + 량) 목록."""

    items: List[FoodEntryForCalories] = Field(
        ...,
        min_length=1,
        description="음식별 이름과 그람수 목록",
    )


class FoodEntryCaloriesResult(BaseModel):
    """항목별 추정 칼로리·영양소."""

    name: str = Field(..., description="음식 이름")
    amount_grams: int = Field(..., description="섭취량 (g)")
    estimated_calories: int = Field(..., description="추정 칼로리 (kcal)")
    nutrients: NutrientInfo = Field(..., description="추정 영양소 (탄수화물/단백질/지방 g)")


class CaloriesCalculationResponse(BaseModel):
    """이름+량 기준 칼로리 계산 결과."""

    items: List[FoodEntryCaloriesResult] = Field(..., description="항목별 추정 칼로리·영양소")
    total_calories: int = Field(..., description="총 추정 칼로리 (kcal)")

