"""FastAPI 入口：识文新裁后端 API。

当前为 S0 脚手架：仅提供 /health 与 /api/chat 占位。
后续阶段接入：
- S2/S3  RAG 检索图（研微问答）
- S4    先贤辩论图
- S5    研究写作图
- S6    新裁角色扮演图
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="识文新裁 API", version="0.1.0")


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest) -> ChatResponse:
    # TODO(S2): 接入 LangGraph 检索图，返回带引据的回答
    return ChatResponse(
        message=f"（占位）研微 RAG 检索图将在 S2 阶段接入。收到问题：{req.query}"
    )
