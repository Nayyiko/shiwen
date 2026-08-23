"""FastAPI 入口：识文新裁后端 API。

S2 已接入 LangGraph 检索图（研微问答）。
后续阶段接入：
- S3    多跳闭环 + 自我反思
- S4    先贤辩论图
- S5    研究写作图
- S6    新裁角色扮演图
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.shiwen.config import get_settings

app = FastAPI(title="识文新裁 API", version="0.2.0")


class ChatRequest(BaseModel):
    query: str
    book_id: str | None = None
    category: str | None = None


class Citation(BaseModel):
    book: str
    chapter: str
    version: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    book_id: str | None = None
    category: str | None = None


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    citation: str
    book_id: str
    book: str
    author: str
    dynasty: str
    chapter: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest) -> ChatResponse:
    """研微 RAG 问答：检索 + 生成，返回带引据的回答。"""
    if not get_settings().deepseek_api_key:
        raise HTTPException(
            status_code=503,
            detail="DEEPSEEK_API_KEY 未配置，请在 .env 中填入 DeepSeek API Key。"
            " 检索功能不受影响，可使用 POST /api/search 测试。",
        )

    from src.shiwen.rag.graph import build_graph

    graph = build_graph()
    result = graph.invoke({
        "query": req.query,
        "book_id": req.book_id,
        "category": req.category,
        "chunks": [],
        "answer": "",
    })

    # 从检索结果中提取引据信息
    citations: list[Citation] = []
    for c in result.get("chunks", []):
        citations.append(Citation(
            book=c["book"],
            chapter=c["chapter"],
            version=c["version"],
            text=c["text"][:200],
        ))

    return ChatResponse(answer=result["answer"], citations=citations)


@app.post("/api/search")
def search(req: SearchRequest) -> SearchResponse:
    """纯检索（不生成）：调试/评测/前端复用，不耗 LLM 费用。"""
    from src.shiwen.rag.retriever import retrieve

    chunks = retrieve(
        query=req.query,
        top_k=req.top_k,
        book_id=req.book_id,
        category=req.category,
    )

    results = [
        SearchResult(
            id=c.id, text=c.text, score=c.score,
            citation=c.citation, book_id=c.book_id,
            book=c.book, author=c.author, dynasty=c.dynasty,
            chapter=c.chapter,
        )
        for c in chunks
    ]
    return SearchResponse(query=req.query, results=results)
