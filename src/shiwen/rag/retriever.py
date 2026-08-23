"""检索层：查询向量化 → Milvus 语义检索 → 带引据元数据的 chunk 列表。

支持元数据前置过滤（book_id / category），后续 S3 多跳闭环会复用此模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.shiwen.ingest.embedder import get_embedder
from src.shiwen.ingest.milvus_store import search as milvus_search


@dataclass
class RetrievedChunk:
    """一个检索结果，携带分数与引据级元数据。"""
    id: str
    text: str
    score: float
    book_id: str
    book: str
    author: str
    dynasty: str
    category: str
    version: str
    part: str | None
    chapter: str
    chapter_index: int
    chunk_index: int

    @property
    def citation(self) -> str:
        """引据短标签：论语·学而篇（通行本）"""
        return f"{self.book}·{self.chapter}（{self.version}）"


def _build_filter(book_id: str | None = None, category: str | None = None) -> str | None:
    """构建 Milvus 标量过滤表达式（元数据前置过滤）。"""
    parts: list[str] = []
    if book_id:
        parts.append(f'book_id == "{book_id}"')
    if category:
        parts.append(f'category == "{category}"')
    return " and ".join(parts) if parts else None


def retrieve(query: str, top_k: int = 5, book_id: str | None = None,
             category: str | None = None) -> list[RetrievedChunk]:
    """语义检索：查询 → 向量化 → Milvus COSINE 检索 → 带元数据的 chunk 列表。

    参数：
        query:    用户原始查询
        top_k:    返回数量（默认 5）
        book_id:  可选，限定某部书（如 "lunyu"）
        category: 可选，限定四部分类（如 "经部"）
    """
    embedder = get_embedder()
    vec = embedder.encode([query])[0]
    filter_expr = _build_filter(book_id=book_id, category=category)

    hits = milvus_search(vec, top_k=top_k, filter_expr=filter_expr)

    results: list[RetrievedChunk] = []
    for h in hits:
        entity = h.get("entity", h)  # search 返回的每个命中可能包在 entity 里
        results.append(RetrievedChunk(
            id=entity["id"],
            text=entity["text"],
            score=h.get("distance", h.get("score", 0.0)),
            book_id=entity["book_id"],
            book=entity["book"],
            author=entity["author"],
            dynasty=entity["dynasty"],
            category=entity["category"],
            version=entity["version"],
            part=entity.get("part") or None,
            chapter=entity["chapter"],
            chapter_index=entity["chapter_index"],
            chunk_index=entity["chunk_index"],
        ))
    return results