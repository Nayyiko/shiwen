"""Ingestion 管线编排：load → chunk → 引据校验(fail-fast) → embed → 写 Milvus + PG。"""

from __future__ import annotations

from pathlib import Path

from . import milvus_store, pg_store, people  # people 用于注册人物表到 Base.metadata
from .chunker import chunk_book
from .embedder import get_embedder
from .models import Chunk, load_manifest

CORPUS_DIR = Path("data/corpus")


def load_book_md(book_id: str) -> str:
    path = CORPUS_DIR / f"{book_id}.md"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，请先运行 normalize")
    return path.read_text(encoding="utf-8")


def validate_chunks(chunks: list[Chunk]) -> None:
    """引据校验：每个 chunk 必须有非空 text / book_id / chapter，且 id 唯一。fail-fast。"""
    seen: set[str] = set()
    problems: list[str] = []
    for c in chunks:
        if not c.text.strip():
            problems.append(f"{c.id}: 空文本")
        if not c.book_id:
            problems.append(f"{c.id}: 缺少 book_id")
        if not c.chapter:
            problems.append(f"{c.id}: 缺少 chapter（引据原子单位）")
        if c.id in seen:
            problems.append(f"{c.id}: id 重复")
        seen.add(c.id)
    if problems:
        raise ValueError("引据校验失败:\n  " + "\n  ".join(problems[:20]))


def run_reindex(book_ids: list[str] | None = None, batch_size: int = 64) -> int:
    """全量重灌：清空 Milvus + PG 后，逐书切分、向量化、入库。返回总 chunk 数。"""
    manifest = load_manifest()
    books = [b for b in manifest.books if book_ids is None or b.id in book_ids]
    if not books:
        print("[!] 没有匹配的书籍")
        return 0

    embedder = get_embedder()
    mv_client = milvus_store._client()
    pg_engine = pg_store.get_engine()

    # 建表 / 建 collection
    pg_store.init_db(pg_engine)
    milvus_store.clear(mv_client)             # 清空（drop 后重建）
    milvus_store.create_collection(mv_client)
    pg_store.clear_chunks(pg_engine)

    total = 0
    for book in books:
        md = load_book_md(book.id)
        chunks = chunk_book(md, book)
        validate_chunks(chunks)

        texts = [c.text for c in chunks]
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(embedder.encode(texts[i:i + batch_size]))

        milvus_store.upsert_chunks(chunks, vectors, mv_client)
        pg_store.insert_chunks(chunks, pg_engine)
        total += len(chunks)
        print(f"[reindex] {book.id:<14} {len(chunks):>5} chunks")

    print(f"[done] 共 {total} chunks（Milvus={milvus_store.count(mv_client)}，PG={pg_store.count(pg_engine)}）")
    return total
