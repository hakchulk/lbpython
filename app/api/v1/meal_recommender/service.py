from app.api.v1.meal_recommender.schemas import (
    CaloriesCalculationRequest,
    CaloriesCalculationResponse,
    DailyMealRecommendation,
    MealRecommendRequest,
)
from app.core.ai_utils import generate


async def recommend_meal(request: MealRecommendRequest) -> DailyMealRecommendation:
    """
    아침·점심·저녁 식단을 한 번에 추천합니다.
    - 목적(체중감량, 건강유지, 근육량증가, 혈당관리, 콜레스테롤)
    - 알러지 (우유, 달걀, 생선, 갑각류, 건과류, 땅콩, 밀)
    - disliked_items: 맘에 들지 않아 전부 새로 받을 때, 제외할 메뉴/재료
    - profile: 회원의 신체/목표/식습관 정보를 포함하면 더욱 개인 맞춤으로 추천
    """

    goal = request.goal.value
    allergies = [a.value for a in request.allergies]

    system_prompt = (
        "당신은 한국인 대상 맞춤 식단을 설계하는 영양사입니다. "
        "항상 사용자의 건강 목표와 알러지를 엄격히 고려해 아침·점심·저녁 세 끼를 한 번에 추천하세요. "
        "추천 메뉴는 한국에서 구하기 쉬운 재료를 중심으로 구성합니다."
    )

    profile = request.profile
    profile_text = ""
    if profile is not None:
        profile_text = (
            "다음은 이 회원의 신체 정보와 목표, 식습관입니다. 반드시 이 정보를 반영해서 식단을 설계하세요.\n"
            f"- 회원 ID: {profile.member_id}, 이름: {profile.name}\n"
            f"- 성별: {profile.gender}, 생년월일: {profile.birthday}\n"
            f"- 키/현재 체중: {profile.height}cm / {profile.weight}kg\n"
            f"- 목표 설명: {profile.goal}, 목표 체중: {profile.goal_weight}kg, 목표 기간(일): {profile.target_date}\n"
            f"- 하루 섭취 목표 칼로리: {profile.daily_calories} kcal\n"
            f"- 회원이 입력한 알러지/피해야 할 음식: {', '.join(profile.allergies) if profile.allergies else '없음'}\n"
            f"- 특이사항/식습관: {profile.special_notes if profile.special_notes else '없음'}\n\n"
        )

    user_prompt = (
        f"다음 조건에 맞는 하루 식단(아침, 점심, 저녁 각 1개씩, 총 3개 메뉴)을 추천해 주세요.\n"
        f"- 식단 목적: {goal}\n"
        f"- 알러지(제외해야 할 식품): {', '.join(allergies) if allergies else '없음'}\n"
        f"- 이번에 제외하고 싶은 음식/재료(disliked_items): "
        f"{', '.join(request.disliked_items) if request.disliked_items else '없음'}\n\n"
        f"{profile_text}"
        "요구사항:\n"
        "1) 아침(breakfast), 점심(lunch), 저녁(dinner) 각각 하나의 대표 메뉴를 꼭 채워 주세요.\n"
        "2) 각 메뉴의 ingredients는 반드시 재료별로 { name: 재료명, grams: 1인분 권장 그람수 } 형태로 넣어 주세요. "
        "   예: [{\"name\": \"닭가슴살\", \"grams\": 150}, {\"name\": \"현미\", \"grams\": 100}]. "
        "   grams는 1인분 기준으로 현실적인 그람수(g)로 적어 주세요.\n"
        "3) 알러지나 disliked_items에 포함된 재료는 어떤 끼니에도 사용하지 마세요.\n"
        "4) 체중감량은 하루 총 칼로리를 적당히 낮추고 포만감 있게, "
        "   근육량증가는 단백질 비율이 높은 메뉴로, 혈당/콜레스테롤 관리는 "
        "   해당 지표를 악화시키는 재료(단순당, 포화지방 등)를 피해서 구성하세요.\n"
        "5) comment 필드에는 오늘 하루 식단을 이렇게 추천한 이유, 목표 달성에 어떻게 도움이 되는지 "
        "   한국어로 자연스럽게 설명해주세요.\n"
        "6) cautions에는 알러지, 혈당/콜레스테롤 관리 측면에서의 주의사항이나 보완 팁을 "
        "   한글 bullet point 느낌의 짧은 문장들로 넣어주세요."
    )

    try:
        response = await generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=DailyMealRecommendation,
        )
        return response.choices[0].message.parsed

    except Exception as e:
        raise ValueError(f"식단 추천 AI 호출 실패: {str(e)}")


async def calculate_calories(
    request: CaloriesCalculationRequest,
) -> CaloriesCalculationResponse:
    """
    유저가 작성한 식단(음식 이름 + 섭취량 g)을 받아
    항목별 추정 칼로리와 영양소(탄수화물/단백질/지방)를 계산합니다.
    """
    items_desc = "\n".join(
        f"- {item.name}: {item.amount_grams}g" for item in request.items
    )

    system_prompt = (
        "당신은 한국 음식의 영양 정보를 추정하는 전문가입니다. "
        "주어진 음식 이름과 섭취량(그람수)에 맞춰 현실적인 칼로리(kcal)와 "
        "탄수화물(carbohydrates), 단백질(protein), 지방(fat) 그람수를 추정해 주세요. "
        "한국에서 흔한 음식 기준으로 답하세요."
    )

    user_prompt = (
        "다음 음식들의 주어진 섭취량(그람수)에 대해 항목별로 "
        "estimated_calories(추정 칼로리 kcal)와 nutrients(탄수화물/단백질/지방 g)를 추정해 주세요.\n\n"
        f"음식 목록:\n{items_desc}\n\n"
        "요구사항:\n"
        "1) items 배열의 순서와 개수를 반드시 입력과 동일하게 유지하세요.\n"
        "2) 각 항목의 name, amount_grams는 입력값 그대로 두고, estimated_calories와 nutrients만 채우세요.\n"
        "3) nutrients는 해당 섭취량 기준 탄수화물/단백질/지방 그람수로, 소수점 가능합니다.\n"
        "4) total_calories는 모든 items의 estimated_calories 합계로 넣어 주세요."
    )

    try:
        response = await generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],            
            temperature=0,
            response_format=CaloriesCalculationResponse,
        )
        return response.choices[0].message.parsed

    except Exception as e:
        raise ValueError(f"칼로리 계산 AI 호출 실패: {str(e)}")


