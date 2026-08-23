"""LangGraph 检索图：三路混合检索 + RRF 融合 + grounding 校验 + 反思多跳闭环。

简历第 1 条的在线实现，G2 Retrieval Graph 的核心。
图结构：retrieve → generate → grounding_check → (pass) END
                                             → (fail) reflect → retrieve (≤3轮)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .bm25_store import search as bm25_search
from .fusion import rrf_fuse
from .generator import generate
from .grounding import check_grounding, reflect
from .query_cleaner import clean_query
from .retriever import RetrievedChunk, retrieve as vector_retrieve


class RAGState(TypedDict, total=False):
    """检索图共享状态。"""
    query: str
    book_id: str | None
    category: str | None
    round: int
    max_rounds: int
    chunks: list[dict]          # RRF 融合后的检索结果
    answer: str
    grounding_pass: bool
    grounding_reason: str
    diagnosis: str
    rewritten_query: str
    trace: list[dict]


def _effective_query(state: RAGState) -> str:
    """当前轮使用的查询：优先改写后的 query，否则原始 query。"""
    rw = state.get("rewritten_query", "")
    return rw if rw else state["query"]


def _retrieve_node(state: RAGState) -> dict:
    """三路混合检索：元数据前置过滤 → 向量 + BM25 并行 → RRF 融合。

    元数据过滤（book_id/category）同时作用于向量和 BM25 两路。
    """
    t0 = time.time()
    # 清洗后的 query 用于检索：剥掉考据/翻译题的提问套话，突出名句内核
    query = clean_query(_effective_query(state))
    book_id = state.get("book_id")
    category = state.get("category")

    # 向量 + BM25 并行检索（两者独立，无依赖）
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_vector = pool.submit(vector_retrieve, query, top_k=20, book_id=book_id, category=category)
        f_bm25 = pool.submit(bm25_search, query, top_k=20, book_id=book_id, category=category)
        vector_hits = [_chunk_to_dict(c) for c in f_vector.result()]
        bm25_hits = f_bm25.result()

    # RRF 融合
    chunks = rrf_fuse(vector_hits, bm25_hits, top_k=5)

    elapsed = round(time.time() - t0, 3)
    trace_entry = {
        "node": "retrieve",
        "elapsed_ms": round(elapsed * 1000),
        "vector_count": len(vector_hits),
        "bm25_count": len(bm25_hits),
        "fused_count": len(chunks),
    }
    return {"chunks": chunks, "trace": state.get("trace", []) + [trace_entry]}


def _generate_node(state: RAGState) -> dict:
    """生成节点：检索结果 + 用户问题 → LLM → 带引据的回答。"""
    t0 = time.time()
    # 从 state dict 重建 RetrievedChunk（过滤 rrf_score 等融合字段）
    _fields = {f.name for f in fields(RetrievedChunk)}
    chunks = [RetrievedChunk(**{k: v for k, v in c.items() if k in _fields})
              for c in state.get("chunks", [])]
    answer = generate(query=state["query"], chunks=chunks)
    elapsed = round(time.time() - t0, 3)
    trace_entry = {
        "node": "generate",
        "elapsed_ms": round(elapsed * 1000),
        "answer_len": len(answer),
    }
    return {"answer": answer, "trace": state.get("trace", []) + [trace_entry]}


def _grounding_node(state: RAGState) -> dict:
    """Grounding 校验：LLM-as-judge 校验回答是否被检索 chunk 支撑 + 引据合规。"""
    t0 = time.time()
    result = check_grounding(state["answer"], state.get("chunks", []))
    elapsed = round(time.time() - t0, 3)
    trace_entry = {
        "node": "grounding_check",
        "elapsed_ms": round(elapsed * 1000),
        "passed": result.passed,
        "unsupported": len(result.unsupported_claims),
    }
    return {
        "grounding_pass": result.passed,
        "grounding_reason": result.reason,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _reflect_node(state: RAGState) -> dict:
    """反思节点：诊断 why miss → 改写查询。"""
    t0 = time.time()
    diagnosis, rewritten = reflect(
        query=state["query"],
        chunks=state.get("chunks", []),
        grounding_reason=state.get("grounding_reason", ""),
    )
    elapsed = round(time.time() - t0, 3)
    trace_entry = {
        "node": "reflect",
        "elapsed_ms": round(elapsed * 1000),
        "diagnosis": diagnosis,
        "rewritten": rewritten[:100],
    }
    return {
        "diagnosis": diagnosis,
        "rewritten_query": rewritten,
        "round": state.get("round", 0) + 1,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _route_grounding(state: RAGState) -> str:
    """条件路由：grounding 通过或轮数耗尽 → 结束；否则 → 反思重检索。"""
    if state.get("grounding_pass", False):
        return "end"
    if state.get("round", 0) >= state.get("max_rounds", 3):
        return "end"
    return "reflect"


def _chunk_to_dict(c: RetrievedChunk) -> dict:
    return {
        "id": c.id, "text": c.text, "score": c.score,
        "book_id": c.book_id, "book": c.book, "author": c.author,
        "dynasty": c.dynasty, "category": c.category, "version": c.version,
        "part": c.part, "chapter": c.chapter,
        "chapter_index": c.chapter_index, "chunk_index": c.chunk_index,
    }


def build_graph() -> StateGraph:
    """构建编译好的检索图（多跳闭环 + 反思）。

    图结构：
        retrieve → generate → grounding_check ──(pass)──> END
                                              ──(fail)──> reflect → retrieve
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("grounding_check", _grounding_node)
    graph.add_node("reflect", _reflect_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grounding_check")

    graph.add_conditional_edges(
        "grounding_check",
        _route_grounding,
        {"end": END, "reflect": "reflect"},
    )
    graph.add_edge("reflect", "retrieve")

    return graph.compile()