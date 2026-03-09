from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def answer_query(question: str, name: str, point: int) -> str:
    try:
        client = OpenAI()
        system_prompt = (
            "당신은 도움이 되는 AI 어시스턴트입니다. "
            "사용자의 질문에 친절하고 쉽게 답변하세요. "
            "주제가 주어지면 그 주제에 맞춰 구체적으로 답변합니다. "
            "파일이나 문서를 참조하지 않고 일반적인 건강, 체력, 식단, 운동 지식을 기반으로 답변하세요. "
            "한줄로 간략하게 답변합니다. "
            "그 외 질문은 '죄송하지만 저는 건강, 체력, 식단, 운동 관련 질문만 답변할 수 있어요.'라고 안내하세요."
            "음식이라는 단어를 포함해서 식단을 추천할 때에 답변해주세요."
        )
        user_prompt = f"사용자 이름: {name}\n현재 포인트: {point}\n질문: {question}"

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
