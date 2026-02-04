from fastapi import FastAPI, Request, status  # Request 추가
from app.api.v1.router import api_router
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI(title="LB Python API")

# 라우터 연결
app.include_router(api_router, prefix="/api/v1")

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "요청하신 페이지(경로)를 찾을 수 없습니다. 주소를 확인해주세요."},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )