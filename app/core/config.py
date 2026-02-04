import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    # 모델명을 여기서 중앙 관리 (기본값 설정)
    AI_MODEL: str = os.getenv("AI_MODEL", "gpt-4o")

settings = Settings()