import os
from openai import AsyncOpenAI
from app.core.config import settings

# 클라이언트 인스턴스화
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def generate(messages: list, response_format, **kwargs):
    """
    OpenAI의 beta.chat.completions.parse API를 호출하는 공통 함수
    **kwargs를 통해 temperature, seed 등 추가 파라미터를 받을 수 있습니다.
    """
    return await client.beta.chat.completions.parse(
        model=settings.AI_MODEL,
        messages=messages,
        response_format=response_format,
        **kwargs  # 호출부에서 넘겨준 temperature=0이 여기로 전달됩니다.
    )