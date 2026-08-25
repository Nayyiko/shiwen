"""FastAPI 入口：识文新裁后端 API。

S2 已接入 LangGraph 多跳检索图（三路混合检索 + RRF + grounding + 反思）。
后续阶段接入：
- S4    先贤辩论图 ✅
- S5    研究写作图 ✅
- S6    新裁角色扮演图 ✅
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.shiwen.config import get_settings

app = FastAPI(title="识文新裁 API", version="0.5.0")


def _sse(data: dict) -> str:
    """格式化为 SSE 事件数据行。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


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


class RoleplayRequest(BaseModel):
    sage_id: str          # kongzi / mengzi / laozi / hanfei
    message: str          # 用户当前消息
    history: list[dict] | None = None  # 对话历史 [{role, content}, ...]
    session_id: str | None = None      # 会话 ID：提供则服务端 Redis 持久化历史，支持断点恢复


class RoleplayResponse(BaseModel):
    sage_id: str
    sage_name: str
    school: str
    response: str
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


@app.post("/api/roleplay")
def roleplay(req: RoleplayRequest) -> RoleplayResponse:
    """新裁角色扮演：1v1 与指定先贤对话，沉浸式叙事。

    S6 Roleplay Graph：检索先贤著作 → 人设生成回复。
    多轮对话由调用方传 history 驱动。
    """
    if not get_settings().deepseek_api_key:
        raise HTTPException(
            status_code=503,
            detail="DEEPSEEK_API_KEY 未配置，请在 .env 中填入 DeepSeek API Key。",
        )

    from src.shiwen.agents.personas import SAGES
    from src.shiwen.memory import MemoryManager
    from src.shiwen.roleplay.graph import build_roleplay_graph

    persona = SAGES.get(req.sage_id)
    if persona is None:
        raise HTTPException(
            status_code=400,
            detail=f"未知先贤 ID：{req.sage_id}。可用：{', '.join(SAGES.keys())}",
        )

    # 分层记忆：session_id 提供时从 Redis 读短期对话历史（断点恢复），否则用调用方传入 history
    memory = MemoryManager()
    history = req.history or []
    if req.session_id:
        stored = memory.get_recent_messages(req.session_id, n=20)
        if stored:
            history = stored

    graph = build_roleplay_graph()
    result = graph.invoke({
        "sage_id": req.sage_id,
        "user_message": req.message,
        "history": history,
        "chunks": [],
        "response": "",
        "trace": [],
    })

    response = result.get("response", "")

    # 状态持久化：把本轮对话写入短期记忆（session_id 提供时），支持下次请求断点恢复
    if req.session_id:
        memory.append_message(req.session_id, "user", req.message)
        memory.append_message(req.session_id, "assistant", response)

    citations = [
        Citation(
            book=c.get("book", ""),
            chapter=c.get("chapter", ""),
            version=c.get("version", ""),
            text=c.get("text", "")[:200],
        )
        for c in result.get("chunks", [])
    ]

    trace = [
        TraceEntry(node=t["node"], elapsed_ms=t.get("elapsed_ms", 0))
        for t in result.get("trace", [])
    ]

    return RoleplayResponse(
        sage_id=req.sage_id,
        sage_name=persona.name,
        school=persona.school,
        response=response,
        citations=citations,
        trace=trace,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SSE 流式端点（四大模块实时输出）
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """研微问答流式：检索 → DeepSeek 逐 token 生成 → 引据。"""
    if not get_settings().deepseek_api_key:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY 未配置")

    def generate():
        from src.shiwen.rag.generator import (
            _SYSTEM_PROMPT, _USER_TEMPLATE, _format_chunks, _build_client)
        from src.shiwen.rag.query_cleaner import clean_query
        from src.shiwen.rag.retriever import retrieve

        s = get_settings()
        client = _build_client()

        # 1. 检索
        query = clean_query(req.query)
        chunks = retrieve(query, top_k=5, book_id=req.book_id, category=req.category)
        citations = [{"book": c.book, "chapter": c.chapter, "version": c.version,
                      "text": c.text[:200]} for c in chunks]
        yield _sse({"type": "citations", "citations": citations})

        # 2. 流式生成
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(
                query=query, chunks_text=_format_chunks(chunks))},
        ]
        stream = client.chat.completions.create(
            model=s.deepseek_model, messages=messages,
            temperature=0.3, max_tokens=1024, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta:
                yield _sse({"type": "token", "token": delta})

        yield _sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/roleplay/stream")
def roleplay_stream(req: RoleplayRequest) -> StreamingResponse:
    """新裁角色扮演流式：检索先贤著作 → DeepSeek 逐 token 生成 → Redis 持久化。"""
    if not get_settings().deepseek_api_key:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY 未配置")

    def generate():
        from openai import OpenAI
        from src.shiwen.agents.personas import SAGES
        from src.shiwen.agents.sage_subgraph import sage_retrieve
        from src.shiwen.context import build_prompt_budget
        from src.shiwen.memory import MemoryManager

        s = get_settings()
        persona = SAGES.get(req.sage_id)
        if persona is None:
            yield _sse({"type": "error", "message": f"未知先贤 ID：{req.sage_id}"})
            return

        memory = MemoryManager()
        history = req.history or []
        if req.session_id:
            stored = memory.get_recent_messages(req.session_id, n=20)
            if stored:
                history = stored

        # 检索 + 上下文治理
        chunks = sage_retrieve(req.message, persona, top_k=3)
        budget = build_prompt_budget(history, chunks)
        history, chunks = budget["history"], budget["chunks"]

        yield _sse({"type": "citations", "citations": [
            {"book": c["book"], "chapter": c["chapter"], "version": c["version"],
             "text": c["text"][:200]} for c in chunks]})

        client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)

        history_text = "\n".join(
            f"【{'用户' if h['role'] == 'user' else persona.name}】{h['content'][:300]}"
            for h in history) if history else "（这是对话的开始）"
        chunks_text = "\n\n".join(
            f"[{i}] {c['text'][:300]}\n    出处：{c['book']}·{c['chapter']}（{c['version']}）"
            for i, c in enumerate(chunks, 1)) if chunks else "（无相关原文，基于学派立场回应）"

        prompt = f"""## 你的身份
你是{persona.name}（{persona.dynasty}时期{persona.school}代表人物）。

## 对话历史
{history_text}

## 用户刚刚说的话
{req.message}

## 你的著作中与话题相关的原文
{chunks_text}

请以{persona.name}的身份回复。要求：严格遵循人设；引用原文标注「书名·篇名（版本）」；自然如对话；100-200 字；不要元描述。"""

        full = ""
        stream = client.chat.completions.create(
            model=s.deepseek_model,
            messages=[{"role": "system", "content": persona.persona_prompt},
                      {"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=400, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta:
                full += delta
                yield _sse({"type": "token", "token": delta})

        if req.session_id:
            memory.append_message(req.session_id, "user", req.message)
            memory.append_message(req.session_id, "assistant", full)

        yield _sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


class DebateInterjectRequest(BaseModel):
    topic_id: str
    message: str


@app.post("/api/write/stream")
def write_stream(req: WriteRequest) -> StreamingResponse:
    """研究写作流式：大纲 → 逐节检索/写作（每节实时推送）→ 综合润色。"""
    if not get_settings().deepseek_api_key:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY 未配置")

    def generate():
        from src.shiwen.writing.graph import (
            _outline_node, _section_retrieve_node, _section_write_node, _synthesize_node)

        state = {"topic": req.topic, "max_sections": req.max_sections,
                 "outline": [], "section_index": 0, "all_chunks": [],
                 "article": "", "trace": []}

        state.update(_outline_node(state))
        outline = state["outline"]
        yield _sse({"type": "outline",
                    "sections": [{"title": s.get("title", "")} for s in outline]})

        while state["section_index"] < len(outline):
            idx = state["section_index"]
            state.update(_section_retrieve_node(state))
            state.update(_section_write_node(state))
            sec = state["outline"][idx]
            yield _sse({"type": "section", "title": sec.get("title", ""),
                        "text": sec.get("text", "")})

        state.update(_synthesize_node(state))
        yield _sse({"type": "article", "article": state.get("article", "")})
        yield _sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/debate/stream")
def debate_stream(req: DebateRequest, topic_id: str = "") -> StreamingResponse:
    """先贤辩论流式：逐轮发言实时推送，支持中途插话（Redis 注入）。"""
    if not get_settings().deepseek_api_key:
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY 未配置")

    def generate():
        from src.shiwen.agents.debate import (
            _topic_parse, _arbitrate_node, _sage_speak_node,
            _drift_check_node, _summarize_node)
        from src.shiwen.redis_store import get_redis

        tid = topic_id or f"debate:{req.topic[:12]}"
        redis = get_redis()
        interj_key = f"debate:{tid}:interjection"

        state = {"topic": req.topic, "user_message": req.user_message or "",
                 "max_speeches": req.max_speeches, "round": 0,
                 "speech_log": [], "last_spoken_round": {},
                 "urgency_trace": [], "drift_events": [], "summary": "", "trace": []}
        state.update(_topic_parse(state))
        yield _sse({"type": "start", "topic": req.topic, "topic_id": tid})

        for _ in range(req.max_speeches):
            # 检查插话（Redis）
            interj = redis.get(interj_key)
            if interj:
                state["user_message"] = interj
                redis.delete(interj_key)
                yield _sse({"type": "interjection", "message": interj})

            state.update(_arbitrate_node(state))
            state.update(_sage_speak_node(state))
            speech = state["speech_log"][-1]
            yield _sse({"type": "speech", "speech": {
                "sage_id": speech["sage_id"], "name": speech["name"],
                "school": speech["school"], "text": speech["text"],
                "urgency_rank": speech.get("urgency_rank", 0),
                "citations": [{"book": c.get("book", ""), "chapter": c.get("chapter", ""),
                               "version": c.get("version", ""), "text": c.get("text", "")[:200]}
                              for c in speech.get("citations", [])],
            }})

            state.update(_drift_check_node(state))

        state.update(_summarize_node(state))
        yield _sse({"type": "summary", "summary": state.get("summary", "")})
        yield _sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/debate/interject")
def debate_interject(req: DebateInterjectRequest) -> dict:
    """辩论插话：把用户中途追问写入 Redis，辩论流下一轮自动注入。"""
    from src.shiwen.redis_store import get_redis
    redis = get_redis()
    redis.set(f"debate:{req.topic_id}:interjection", req.message, ttl=600)
    return {"ok": True}
