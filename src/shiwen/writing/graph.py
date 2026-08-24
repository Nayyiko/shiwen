"""研究写作图（LangGraph StateGraph）。

S5 Writing Graph：选题→大纲→逐节检索→逐节写作→综合润色，可断点续写。
图结构：
    outline → section_router → section_retrieve → section_write → section_router
    section_router → (all done) → synthesize → END
"""

from __future__ import annotations

import json
import re
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from src.shiwen.config import get_settings
from src.shiwen.rag.bm25_store import search as bm25_search
from src.shiwen.rag.fusion import rrf_fuse
from src.shiwen.rag.retriever import RetrievedChunk, retrieve as vector_retrieve

from .citations import chunk_citation_label


class WritingState(TypedDict, total=False):
    """写作图共享状态。"""
    topic: str
    max_sections: int
    outline: list[dict]        # [{title, points, query, chunks, text, done}]
    section_index: int          # 当前正在处理的节索引
    all_chunks: list[dict]      # 全部检索 chunk（用于引据校验）
    article: str                # 最终文章
    trace: list[dict]


# ── 大纲生成 ─────────────────────────────────────────────────────────────────

_OUTLINE_PROMPT = """你是研微（YanWei），一位古籍研究助手。请为以下研究选题生成一份结构化大纲。

## 选题
{topic}

## 要求
1. 生成 {max_sections} 节，每节一个小标题 + 2-3 个要点 + 1 个检索关键词（用于从古籍语料库中检索原文支撑）。
2. 大纲应覆盖选题的主要方面，逻辑递进（从概念辨析到文本分析到当代意义）。
3. 返回纯 JSON 数组（不要 markdown 代码块标记）：
```json
[
  {{"title": "第一节标题", "points": ["要点1", "要点2"], "query": "检索关键词"}},
  ...
]
```"""


def _outline_node(state: WritingState) -> dict:
    """选题理解 → LLM 生成大纲。"""
    t0 = time.time()
    s = get_settings()
    client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)

    max_sec = state.get("max_sections", 4)
    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "user", "content": _OUTLINE_PROMPT.format(
            topic=state["topic"], max_sections=max_sec)}],
        temperature=0.3,
        max_tokens=1024,
    )
    content = response.choices[0].message.content or "[]"

    # 解析：支持裸 JSON 或 markdown 代码块
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        outline = json.loads(content)
    except json.JSONDecodeError:
        # 容错：尝试提取 [...] 部分
        m = re.search(r"\[.*\]", content, re.DOTALL)
        outline = json.loads(m.group(0)) if m else []

    if not isinstance(outline, list) or not outline:
        outline = [{"title": "概述", "points": ["总论"], "query": state["topic"]}]

    # 初始化每节状态
    for item in outline:
        item.setdefault("chunks", [])
        item.setdefault("text", "")
        item.setdefault("done", False)

    trace_entry = {"node": "outline", "elapsed_ms": round((time.time() - t0) * 1000),
                   "sections": len(outline)}
    return {"outline": outline, "section_index": 0, "all_chunks": [],
            "trace": [trace_entry]}


# ── 逐节检索 ────────────────────────────────────────────────────────────────

def _section_retrieve_node(state: WritingState) -> dict:
    """第 i 节：三路混合检索（复用现有检索层）。"""
    t0 = time.time()
    idx = state.get("section_index", 0)
    outline = state.get("outline", [])
    if idx >= len(outline):
        return {}

    section = outline[idx]
    query = section.get("query", state["topic"])

    # 向量 + BM25 并行
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fv = pool.submit(vector_retrieve, query, top_k=5)
        fb = pool.submit(bm25_search, query, top_k=5)
        vh = [_chunk_to_dict(c) for c in fv.result()]
        bh = fb.result()
    chunks = rrf_fuse(vh, bh, top_k=5)

    section["chunks"] = chunks

    all_chunks = state.get("all_chunks", [])
    seen_ids = {c["id"] for c in all_chunks}
    for c in chunks:
        if c["id"] not in seen_ids:
            seen_ids.add(c["id"])
            all_chunks.append(c)

    trace_entry = {"node": "section_retrieve", "elapsed_ms": round((time.time() - t0) * 1000),
                   "section": idx, "chunks": len(chunks)}

    return {"outline": outline, "all_chunks": all_chunks,
            "trace": state.get("trace", []) + [trace_entry]}


# ── 逐节写作 ────────────────────────────────────────────────────────────────

_SECTION_WRITE_PROMPT = """你是研微（YanWei），一位古籍研究助手。请为以下研究文章撰写一节正文。

## 研究选题
{topic}

## 本节标题
{section_title}

## 本节要点
{points}

## 检索资料（必须引用）
{chunks_text}

## 写作要求
1. 写 200-400 字，紧扣本节标题与要点。
2. **每引用一条原文，必须标注出处**，格式为「书名·篇名（版本）」，例如「论语·学而篇第一（通行本）」。篇名必须使用检索资料中出现的完整名称（含序号），不可省略"第一""上""下"等后缀。引据必须来自上面的检索资料，不可编造。
3. 语言风格：学术严谨但可读，适合非专业读者。
4. 只写本节正文，不要写"本节将讨论..."之类的元描述。"""


def _section_write_node(state: WritingState) -> dict:
    """第 i 节：LLM 写作 + 引据标注。"""
    t0 = time.time()
    s = get_settings()
    client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)

    idx = state.get("section_index", 0)
    outline = state.get("outline", [])
    section = outline[idx]

    chunks_text = "\n\n".join(
        f"[{i}] {c['text'][:300]}\n    出处：{chunk_citation_label(c)}"
        for i, c in enumerate(section.get("chunks", []), 1)
    ) if section.get("chunks") else "（无相关原文，请基于已有知识写，但不要编造具体引文）"

    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "user", "content": _SECTION_WRITE_PROMPT.format(
            topic=state["topic"],
            section_title=section["title"],
            points="、".join(section.get("points", [])),
            chunks_text=chunks_text,
        )}],
        temperature=0.4,
        max_tokens=600,
    )

    section["text"] = response.choices[0].message.content or ""
    section["done"] = True
    next_idx = idx + 1

    trace_entry = {"node": "section_write", "elapsed_ms": round((time.time() - t0) * 1000),
                   "section": idx, "text_len": len(section["text"])}

    return {"outline": outline, "section_index": next_idx,
            "trace": state.get("trace", []) + [trace_entry]}


# ── 综合润色 ────────────────────────────────────────────────────────────────

_SYNTHESIZE_PROMPT = """你是研微（YanWei），一位古籍研究助手。请将以下各节正文综合为一篇完整的研究文章。

## 选题
{topic}

## 各节正文
{sections_text}

## 检索资料总览（已用于各节写作）
{chunks_overview}

## 要求
1. 撰写一段引言（100-150 字），引出选题。
2. 将各节正文按顺序拼接，确保过渡自然。不要修改各节已有的引据标注。
3. 撰写一段结语（100-150 字），总结全文要点。
4. 在文末附加「## 引据清单」，列出全文引用的所有书籍（去重，按引用次数降序）。
5. 全文格式：Markdown，引言用 `> ` 引用块，各节用 `## 节标题`。"""


def _synthesize_node(state: WritingState) -> dict:
    """综合润色：引言 + 合并各节 + 结语 + 引据清单。"""
    t0 = time.time()
    s = get_settings()
    client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)

    outline = state.get("outline", [])
    sections_text = "\n\n".join(
        f"## {sec['title']}\n{sec['text']}"
        for sec in outline if sec.get("text")
    )

    all_chunks = state.get("all_chunks", [])
    chunks_overview = "\n".join(
        f"- {chunk_citation_label(c)}"
        for c in all_chunks[:10]
    ) if all_chunks else "（无）"

    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "user", "content": _SYNTHESIZE_PROMPT.format(
            topic=state["topic"],
            sections_text=sections_text,
            chunks_overview=chunks_overview,
        )}],
        temperature=0.3,
        max_tokens=1500,
    )

    article = response.choices[0].message.content or ""

    trace_entry = {"node": "synthesize", "elapsed_ms": round((time.time() - t0) * 1000),
                   "article_len": len(article)}

    return {"article": article, "trace": state.get("trace", []) + [trace_entry]}


# ── 条件路由 ────────────────────────────────────────────────────────────────


def _section_router(state: WritingState) -> str:
    """条件路由：还有未完成的节 → 继续检索；全部完成 → 综合。"""
    idx = state.get("section_index", 0)
    outline = state.get("outline", [])
    if idx < len(outline):
        return "retrieve"
    return "synthesize"


# ── 构建图 ──────────────────────────────────────────────────────────────────


def build_writing_graph() -> StateGraph:
    """构建编译好的写作图。

    图结构：
        outline → section_router → section_retrieve → section_write → section_router
        section_router → (all done) → synthesize → END
    """
    graph = StateGraph(WritingState)

    graph.add_node("outline", _outline_node)
    graph.add_node("section_retrieve", _section_retrieve_node)
    graph.add_node("section_write", _section_write_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("outline")
    graph.add_edge("outline", "section_retrieve")
    graph.add_edge("section_retrieve", "section_write")

    graph.add_conditional_edges(
        "section_write",
        _section_router,
        {"retrieve": "section_retrieve", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


def _chunk_to_dict(c: RetrievedChunk) -> dict:
    return {
        "id": c.id, "text": c.text, "score": c.score,
        "book_id": c.book_id, "book": c.book, "author": c.author,
        "dynasty": c.dynasty, "category": c.category, "version": c.version,
        "part": c.part, "chapter": c.chapter,
        "chapter_index": c.chapter_index, "chunk_index": c.chunk_index,
    }