from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv(override=True)
# 로거 설정
logger = logging.getLogger("uvicorn.error") # uvicorn 로그와 통합

def answer_query(question: str, name: str, point: int, user_info: str = "") -> str:
    try:
        client = OpenAI()
        system_prompt = (
            "당신은 사용자의 신체 정보와 목표를 분석하여 최적의 솔루션을 제공하는 'Last Layer 건강 비서'입니다. "
            "반드시 다음 원칙에 따라 답변하십시오.\n\n"
            "답변 내용 중 사용자의 정보를 참조 한 항목이 있으면 그 항목만 답변 시작 시 자연스런 문장으로 보여주며 대화를 시작하십시오.\n"
            "답변 범위 제한: '건강', '체력', '식단', '운동' 관련 질문에만 전문적으로 답변하십시오.\n"
            "그 외의 주제(정치, 경제, 일반 상식 등)에 대해서는 '건강 관리와 관련된 질문에만 답변을 드릴 수 있습니다'라고 정중히 거절하십시오.\n"
            "식단 추천 시 사용자의 [알레르기 정보]를 확인하여 해당 재료가 포함된 메뉴는 절대 제외하십시오.\n"
            "모든 답변은 한국어로 친절하고 신뢰감 있는 전문가 어조로 작성하십시오."
            "답변은 2문장 이하로 간략하게 답변합니다."
        )
        user_prompt = f"질문: {question}"
        user_prompt += f"\n{user_info}"
        logger.info(f"{user_info}")
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.2,
        )

        answer = res.choices[0].message.content
        return answer or "죄송합니다. 답변을 생성하지 못했습니다."
    except Exception as e:
        print("AI 응답 오류:", e)
        return "서버 오류가 발생했습니다."
