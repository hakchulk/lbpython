from fastapi import APIRouter, File, UploadFile, HTTPException

from .service import analyze_diet_image
from .schemas import DietAnalysisResponse

# 1. 라우터 생성 (prefix와 tags는 v1/router.py에서 통합 관리하므로 여기선 생략 가능)
router = APIRouter()


@router.post("/analyze", response_model=DietAnalysisResponse)
async def upload_diet_image(file: UploadFile = File(...)):
    """
    사용자가 업로드한 식단 이미지를 분석합니다.
    """

    # 1. 파일 자체가 비어있는지 혹은 파일명이 없는지 체크
    if not file or not file.filename:
        raise HTTPException(
            status_code=400,
            detail="업로드된 파일이 없습니다.",
        )

    # 2. content_type이 None이거나 이미지로 시작하지 않는지 체크 (핵심 수정!)
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일(jpg, png 등)만 업로드할 수 있으며, 파일 형식을 확인할 수 없습니다.",
        )

    try:
        image_data = await file.read()

        # 3. 읽은 데이터가 0바이트인지 체크
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="파일의 내용이 비어있습니다.")

        result = await analyze_diet_image(image_data)
        return result

    except Exception as e:
        # 에러 로그 출력 (디버깅용)
        print(f"Error during analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail="식단 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )
    finally:
        await file.close()