import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    AI_MODEL: str = os.getenv("AI_MODEL", "gpt-4o")


# 인스턴스 생성
settings = Settings()


def _validate_settings() -> None:
    """앱 기동 시 필수 환경 변수 검사. 누락 시 즉시 실패하여 첫 요청에서의 에러를 방지."""
    if not settings.OPENAI_API_KEY or not str(settings.OPENAI_API_KEY).strip():
        raise ValueError(
            "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일 또는 환경 변수를 확인하세요."
        )


_validate_settings()