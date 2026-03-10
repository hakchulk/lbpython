import io
import base64
from PIL import Image

from .schemas import DietAnalysisResponse
from app.core.ai_utils import generate  # 범용 유틸리티 임포트

def process_and_encode_image(image_bytes: bytes) -> str:
    """이미지 크기를 512x512 이내로 리사이징하고 base64로 인코딩합니다."""
    image = Image.open(io.BytesIO(image_bytes))
    
    # [리사이징 로직]
    # thumbnail은 원본 비율을 유지하며, 가로/세로 중 긴 쪽을 1024에 맞춥니다.
    # 이미 1024보다 작다면 변경하지 않습니다.
    max_size = (512, 512)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # 리사이징된 이미지를 바이트로 변환
    buffer = io.BytesIO()
    # 원본 포맷 유지 (알 수 없는 경우 JPEG로 저장)
    img_format = image.format if image.format else "JPEG"
    image.save(buffer, format=img_format)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

async def analyze_diet_image(image_bytes: bytes) -> DietAnalysisResponse:
    # 이미지 최적화 및 인코딩
    base64_image = process_and_encode_image(image_bytes)

    prompt = "이 음식 사진을 분석해서 칼로리,영양 성분 정보, 식단에 대한 평가 제공해줘."

    try:
        response = await generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low" # 비용 절감을 원할 경우 "low" 설정 가능
                            }
                        },
                    ],
                }
            ],            
            temperature=0,
            response_format=DietAnalysisResponse,
        )
        return response.choices[0].message.parsed
    
    except Exception as e:
        raise ValueError(f"AI 분석 실패: {str(e)}")