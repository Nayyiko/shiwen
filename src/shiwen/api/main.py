"""FastAPI 入口：识文新裁后端 API。

S2 已接入 LangGraph 多跳检索图（三路混合检索 + RRF + grounding + 反思）。
后续阶段接入：
- S4    先贤辩论图 ✅
- S5    研究写作图 ✅
- S6    新裁角色扮演图
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.shiwen.config import get_settings

app = FastAPI(title="识文新裁 API", version="0.5.0")


class ChatRequest(BaseModel):
    query: str
    book_id: str | None = None
    category: str | None = None


class Citation(BaseModel):
    book: str
    chapter: str
    version: str
    text: str


class TraceEntry(BaseModel):
    node: str
    elapsed_ms: int


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    rounds: int
    grounding_pass: bool
    grounding_reason: str | None = None
    diagnosis: str | None = None
    trace: list[TraceEntry]


class DebateRequest(BaseModel):
    topic: str
    user_message: str | None = None
    max_speeches: int = 8


class DebateSpeech(BaseModel):
    sage_id: str
    name: str
    school: str
    text: str
    citations: list[Citation]
    urgency_rank: int


class UrgencyRanking(BaseModel):
    sage_id: str
    name: str
    total: float
    relevance: float
    recency: float
    rank: int


class DebateResponse(BaseModel):
    topic: str
    speeches: list[DebateSpeech]
    summary: str
    urgency_trace: list[dict]
    drift_events: list[dict]
    trace: list[TraceEntry]


class WriteRequest(BaseModel):
    topic: str
    max_sections: int = 4


class WriteSection(BaseModel):
    title: str
    text: str
    citations_count: int


class WriteResponse(BaseModel):
    topic: str
    sections: list[WriteSection]
    article: str
    citations: list[Citation]
    trace: list[TraceEntry]


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
    """研微 RAG 问答：三路混合检索 → RRF 融合 → 生成 → grounding 校验 → 反思多跳（≤3轮）。"""
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
        "round": 0,
        "max_rounds": 3,
        "chunks": [],
        "answer": "",
        "grounding_pass": False,
        "grounding_reason": "",
        "diagnosis": "",
        "rewritten_query": "",
        "trace": [],
    })

    citations: list[Citation] = []
    for c in result.get("chunks", []):
        citations.append(Citation(
            book=c["book"],
            chapter=c["chapter"],
            version=c["version"],
            text=c["text"][:200],
        ))

    trace = [
        TraceEntry(node=t["node"], elapsed_ms=t["elapsed_ms"])
        for t in result.get("trace", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        citations=citations,
        rounds=result.get("round", 0) + 1,
        grounding_pass=result.get("grounding_pass", False),
        grounding_reason=result.get("grounding_reason") or None,
        diagnosis=result.get("diagnosis") or None,
        trace=trace,
    )


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


@app.post("/api/debate")
def debate(req: DebateRequest) -> DebateResponse:
    """先贤辩论：多先贤紧急度仲裁 → 轮流发言 → 人格漂移监测 → 主持总结。

    S4 Debate Graph：每位先贤 = 人设 prompt + RAG 子图（限定本人著作检索）。
    """
    if not get_settings().deepseek_api_key:
        raise HTTPException(
            status_code=503,
            detail="DEEPSEEK_API_KEY 未配置，请在 .env 中填入 DeepSeek API Key。",
        )

    from src.shiwen.agents.debate import build_debate_graph

    graph = build_debate_graph()
    result = graph.invoke({
        "topic": req.topic,
        "user_message": req.user_message or "",
        "max_speeches": req.max_speeches,
        "round": 0,
        "speech_log": [],
        "last_spoken_round": {},
        "urgency_trace": [],
        "drift_events": [],
        "summary": "",
        "trace": [],
    })

    speeches: list[DebateSpeech] = []
    for h in result.get("speech_log", []):
        citations = [
            Citation(
                book=c.get("book", ""),
                chapter=c.get("chapter", ""),
                version=c.get("version", ""),
                text=c.get("text", "")[:200],
            )
            for c in h.get("citations", [])
        ]
        speeches.append(DebateSpeech(
            sage_id=h["sage_id"],
            name=h["name"],
            school=h["school"],
            text=h["text"],
            citations=citations,
            urgency_rank=h.get("urgency_rank", 0),
        ))

    trace = [
        TraceEntry(node=t["node"], elapsed_ms=t.get("elapsed_ms", 0))
        for t in result.get("trace", [])
    ]

    return DebateResponse(
        topic=req.topic,
        speeches=speeches,
        summary=result.get("summary", ""),
        urgency_trace=result.get("urgency_trace", []),
        drift_events=result.get("drift_events", []),
        trace=trace,
    )


@app.post("/api/write")
def write(req: WriteRequest) -> WriteResponse:
    """研究写作：选题→大纲→逐节检索→逐节写作→综合润色。

    S5 Writing Graph：LangGraph 状态图，可断点续写。
    """
    if not get_settings().deepseek_api_key:
        raise HTTPException(
            status_code=503,
            detail="DEEPSEEK_API_KEY 未配置，请在 .env 中填入 DeepSeek API Key。",
        )

    from src.shiwen.writing.graph import build_writing_graph

    graph = build_writing_graph()
    result = graph.invoke({
        "topic": req.topic,
        "max_sections": req.max_sections,
        "outline": [],
        "section_index": 0,
        "all_chunks": [],
        "article": "",
        "trace": [],
    })

    sections: list[WriteSection] = []
    for sec in result.get("outline", []):
        sections.append(WriteSection(
            title=sec.get("title", ""),
            text=sec.get("text", ""),
            citations_count=len(sec.get("chunks", [])),
        ))

    citations = [
        Citation(
            book=c.get("book", ""),
            chapter=c.get("chapter", ""),
            version=c.get("version", ""),
            text=c.get("text", "")[:200],
        )
        for c in result.get("all_chunks", [])[:10]
    ]

    trace = [
        TraceEntry(node=t["node"], elapsed_ms=t.get("elapsed_ms", 0))
        for t in result.get("trace", [])
    ]

    return WriteResponse(
        topic=req.topic,
        sections=sections,
        article=result.get("article", ""),
        citations=citations,
        trace=trace,
    )
