"""LangGraph 检索图：单跳 RAG（retrieve → generate）。

这是整个系统的共用检索基座——一处构建，三处（问答/辩论/写作）复用。
S2 为单跳；S3 将在此基础上升级为多跳闭环（反思 → 改写 → 再检索）。
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .generator import generate
from .retriever import RetrievedChunk, retrieve


class RAGState(TypedDict):
    """检索图共享状态：查询 → 检索结果 → 生成回答。"""
    query: str
    book_id: str | None
    category: str | None
    chunks: list[dict]          # RetrievedChunk 序列化后存入
    answer: str


def _retrieve_node(state: RAGState) -> dict:
    """检索节点：查询 → 向量化 → Milvus 检索 → 返回 chunk 列表。"""
    chunks = retrieve(
        query=state["query"],
        top_k=5,
        book_id=state.get("book_id"),
        category=state.get("category"),
    )
    return {"chunks": [_chunk_to_dict(c) for c in chunks]}


def _generate_node(state: RAGState) -> dict:
    """生成节点：检索结果 + 用户问题 → LLM → 带引据的回答。"""
    chunks = [_dict_to_chunk(d) for d in state.get("chunks", [])]
    answer = generate(query=state["query"], chunks=chunks)
    return {"answer": answer}


def _chunk_to_dict(c: RetrievedChunk) -> dict:
    return {
        "id": c.id, "text": c.text, "score": c.score,
        "book_id": c.book_id, "book": c.book, "author": c.author,
        "dynasty": c.dynasty, "category": c.category, "version": c.version,
        "part": c.part, "chapter": c.chapter,
        "chapter_index": c.chapter_index, "chunk_index": c.chunk_index,
    }


def _dict_to_chunk(d: dict) -> RetrievedChunk:
    return RetrievedChunk(**d)


def build_graph() -> StateGraph:
    """构建编译好的检索图（单跳）。

    图结构：retrieve → generate → END
    """
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()