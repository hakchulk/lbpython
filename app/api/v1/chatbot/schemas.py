from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="사용자 질문")
    userInfo: str = Field(..., description="사용자 정보")


class ChatResponse(BaseModel):
    question: str = Field(..., description="요청한 질문")
    answer: str = Field(..., description="챗봇 응답")
