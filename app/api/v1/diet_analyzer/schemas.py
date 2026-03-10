from pydantic import BaseModel, Field
from typing import List, Literal

class NutrientInfo(BaseModel):
    carbohydrates: float = Field(..., description="탄수화물 (g)")
    protein: float = Field(..., description="단백질 (g)")
    fat: float = Field(..., description="지방 (g)")


class FoodItem(BaseModel):
    name: str = Field(..., description="음식 명칭 (예: 닭가슴살, 흰쌀밥)")
    weight_gram: int = Field(..., description="예상 무게 (g)")
    calories: int = Field(..., description="해당 음식의 예상 칼로리 (kcal)")

class DietAnalysisResponse(BaseModel):
    food_name: str = Field(..., description="분석된 음식 이름")
    calories: int = Field(..., description="예상 총 칼로리 (kcal)")
    items: List[FoodItem] = Field(..., description="사진에서 포착된 개별 음식 리스트")
    nutrients: NutrientInfo = Field(..., description="상세 영양 성분")
    evaluation: str = Field(..., description="식단에 대한 평가")
    status: Literal["SUCCESS", "PARTIAL_SUCCESS", "NO_FOOD_DETECTED", "INVALID_IMAGE", "ERROR"] = Field(
        default="SUCCESS", 
        description="분석 상태 (성공, 부분성공, 음식없음, 이미지오류, 서버오류)"
    )