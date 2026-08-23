"""BM25 关键词检索：jieba 分词 + rank_bm25，进程内懒建索引。

向量检索擅长语义，BM25 擅长精确古词匹配——二者互补，RRF 融合后显著提升召回。
"""

from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi

from src.shiwen.ingest import pg_store
from src.shiwen.ingest.pg_store import ChunkRow

# 进程级缓存：首次检索时从 PG 加载全量 chunk 建索引，后续复用。
_INDEX: BM25Okapi | None = None
_CHUNKS: list[dict] = []  # 与 _INDEX 同步的 chunk 元数据列表


def _chunk_row_to_dict(row: ChunkRow) -> dict:
    return {
        "id": row.id, "text": row.text, "book_id": row.book_id,
        "book": row.book, "author": row.author, "dynasty": row.dynasty,
        "category": row.category, "version": row.version,
        "part": row.part, "chapter": row.chapter,
        "chapter_index": row.chapter_index, "chunk_index": row.chunk_index,
    }


def _ensure_index() -> None:
    global _INDEX, _CHUNKS
    if _INDEX is not None:
        return
    engine = pg_store.get_engine()
    from sqlalchemy.orm import Session
    with Session(engine) as session:
        rows = session.query(ChunkRow).all()
    _CHUNKS = [_chunk_row_to_dict(r) for r in rows]
    corpus = [list(jieba.cut(c["text"])) for c in _CHUNKS]
    _INDEX = BM25Okapi(corpus)


def clear_index() -> None:
    """reindex 后调用，使缓存失效。"""
    global _INDEX, _CHUNKS
    _INDEX = None
    _CHUNKS = []


def search(query: str, top_k: int = 20, book_id: str | None = None,
           category: str | None = None) -> list[dict]:
    """BM25 关键词检索，支持元数据过滤。

    返回：
        [{"id": ..., "text": ..., "score": ..., "book_id": ..., ...}, ...]
    """
    _ensure_index()
    tokens = list(jieba.cut(query))
    scores = _INDEX.get_scores(tokens)

    # 构造带分数的 (idx, score) 列表，过滤后排序
    candidates: list[tuple[int, float]] = []
    for i, score in enumerate(scores):
        if score <= 0:
            continue
        c = _CHUNKS[i]
        if book_id and c["book_id"] != book_id:
            continue
        if category and c["category"] != category:
            continue
        candidates.append((i, score))

    candidates.sort(key=lambda x: -x[1])
    results: list[dict] = []
    for idx, score in candidates[:top_k]:
        row = dict(_CHUNKS[idx])
        row["score"] = float(score)
        results.append(row)
    return results