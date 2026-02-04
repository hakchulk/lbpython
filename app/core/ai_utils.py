import os
from openai import AsyncOpenAI
from app.core.config import settings

# 클라이언트 인스턴스화
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def generate(messages: list, response_format):
    """OpenAI의 beta.chat.completions.parse API를 호출하는 공통 함수"""
    return await client.beta.chat.completions.parse(
        model=settings.AI_MODEL,
        messages=messages,
        response_format=response_format
    )