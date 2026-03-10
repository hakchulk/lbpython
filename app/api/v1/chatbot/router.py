from fastapi import APIRouter

from app.api.v1.chatbot.chatlangchain import answer_query
from app.api.v1.chatbot.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

@router.post("", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        answer = answer_query(req.question, req.userInfo)
        return {"question": req.question, "answer": answer or "죄송합니다. 답변을 생성하지 못했습니다."}
    except Exception as e:
        print("챗봇 처리 오류:", e)
        return {"question": req.question, "answer": "서버 오류가 발생했습니다."}
