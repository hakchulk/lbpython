import io
import os
from google import genai
from pydantic import ValidationError
from dotenv import load_dotenv
from PIL import Image

from .schemas import DietAnalysisResponse

# .env 파일의 내용을 환경 변수로 로드
load_dotenv()

# 환경 변수에서 키 가져오기
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AI_MODEL=os.getenv("AI_MODEL")
# 클라이언트 초기화 (최신 google-genai 방식)
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("에러: .env 파일에서 GOOGLE_API_KEY를 찾을 수 없습니다.")

async def analyze_diet_image(image_bytes: bytes) -> DietAnalysisResponse:
    """
    이미지 데이터를 받아 최신 Google GenAI SDK를 통해 분석하고 
    구조화된 Pydantic 모델로 반환합니다.
    """
    # [이미지 처리] 바이너리를 PIL 객체로 변환
    image = Image.open(io.BytesIO(image_bytes))

    # [프롬프트 정의]
    prompt = "이 음식 사진을 분석해서 칼로리와 영양 성분 정보를 제공해줘."

    try:
        # [AI 호출] 최신 SDK 방식 적용
        # response_mime_type을 지정하면 AI가 순수 JSON만 반환합니다.
        response = client.models.generate_content(
            model=AI_MODEL,
            contents=[prompt, image],
            config={
                "response_mime_type": "application/json",
                "response_schema": DietAnalysisResponse, # 스키마를 직접 전달 가능
            }
        )
        
        # [결과 파싱] response.parsed를 사용하면 자동으로 Pydantic 모델로 변환됩니다.
        # (만약 response_schema를 쓰지 않는다면 response.text를 json.loads 하면 됩니다.)
        return response.parsed
    
    except ValidationError as ve:
        raise ValueError(f"데이터 규격 불일치: {str(ve)}")
    except Exception as e:
        raise ValueError(f"AI 응답 분석 실패: {str(e)}")