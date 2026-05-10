# api.py
"""FastAPI REST API 接口"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service.rag_service import RAGService

app = FastAPI(title="金融风控 RAG API")

# 全局单例 RAG 服务
rag_service = None


def get_rag_service():
    """获取 RAG 服务实例（延迟初始化）"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service


class QuestionRequest(BaseModel):
    """问答请求"""
    question: str


class AnswerResponse(BaseModel):
    """问答响应"""
    answer: str
    source_db: str = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化 RAG 服务"""
    get_rag_service()


@app.get("/")
def root():
    """根路径"""
    return {"message": "金融风控 RAG API 已启动", "version": "1.0.0"}


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    """问答接口"""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        service = get_rag_service()
        answer = service.ask(req.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
