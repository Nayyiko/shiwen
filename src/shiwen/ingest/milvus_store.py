"""Milvus 向量库层：collection `shiwen_chunks`。

仅依赖 Settings.milvus_uri（默认回落 milvus_db_path 的 Milvus Lite 文件模式），
未来换 standalone Milvus 只改配置不改代码。索引用 FLAT + COSINE（Lite 下最稳）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pymilvus import DataType, MilvusClient

from src.shiwen.config import get_settings
from .models import Chunk

COLLECTION = "shiwen_chunks"
DIM = 1024

# 输出字段（search/query 时带回元数据）
_OUTPUT_FIELDS = [
    "id", "text", "book_id", "book", "author", "dynasty",
    "category", "version", "part", "chapter", "chapter_index", "chunk_index",
]


@lru_cache(maxsize=1)
def _client() -> MilvusClient:
    """Milvus Lite 客户端（进程内单例，避免频繁创建连接导致 gRPC 过载）。"""
    s = get_settings()
    return MilvusClient(uri=s.milvus_uri or s.milvus_db_path)


def create_collection(client: MilvusClient | None = None) -> None:
    client = client or _client()
    if client.has_collection(COLLECTION):
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=16384)
    schema.add_field("book_id", DataType.VARCHAR, max_length=128)
    schema.add_field("book", DataType.VARCHAR, max_length=128)
    schema.add_field("author", DataType.VARCHAR, max_length=128)
    schema.add_field("dynasty", DataType.VARCHAR, max_length=64)
    schema.add_field("category", DataType.VARCHAR, max_length=64)
    schema.add_field("version", DataType.VARCHAR, max_length=64)
    schema.add_field("part", DataType.VARCHAR, max_length=256)
    schema.add_field("chapter", DataType.VARCHAR, max_length=256)
    schema.add_field("chapter_index", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    client.create_collection(COLLECTION, schema=schema, index_params=index_params)


def _to_row(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "vector": vector,
        "text": chunk.text,
        "book_id": chunk.book_id,
        "book": chunk.book,
        "author": chunk.author,
        "dynasty": chunk.dynasty,
        "category": chunk.category,
        "version": chunk.version,
        "part": chunk.part or "",
        "chapter": chunk.chapter or "",
        "chapter_index": chunk.chapter_index,
        "chunk_index": chunk.chunk_index,
    }


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]],
                  client: MilvusClient | None = None) -> None:
    client = client or _client()
    if not chunks:
        return
    rows = [_to_row(c, v) for c, v in zip(chunks, vectors)]
    client.insert(collection_name=COLLECTION, data=rows)


def _ensure_loaded(client: MilvusClient) -> None:
    """Milvus Lite 在新进程里集合处于 released 状态，search/query 前必须先 load。

    load 幂等：已加载的集合重复调用无副作用。
    """
    if client.has_collection(COLLECTION):
        client.load_collection(COLLECTION)


def search(query_vector: list[float], top_k: int = 5,
           filter_expr: str | None = None, client: MilvusClient | None = None) -> list[dict]:
    client = client or _client()
    _ensure_loaded(client)
    res = client.search(
        collection_name=COLLECTION,
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=_OUTPUT_FIELDS,
    )
    return res[0] if res else []


def query(filter_expr: str, limit: int = 100, client: MilvusClient | None = None) -> list[dict]:
    client = client or _client()
    _ensure_loaded(client)
    return client.query(
        collection_name=COLLECTION,
        filter=filter_expr,
        output_fields=_OUTPUT_FIELDS,
        limit=limit,
    )


def count(client: MilvusClient | None = None) -> int:
    client = client or _client()
    if not client.has_collection(COLLECTION):
        return 0
    stats = client.get_collection_stats(COLLECTION)
    return int(stats.get("row_count", 0))


def count_where(filter_expr: str, client: MilvusClient | None = None) -> int:
    """按过滤条件计数（get_collection_stats 只有总数，得回退到 query + len）。"""
    client = client or _client()
    if not client.has_collection(COLLECTION):
        return 0
    _ensure_loaded(client)
    res = client.query(collection_name=COLLECTION, filter=filter_expr,
                       output_fields=["id"], limit=100000)
    return len(res)


def clear(client: MilvusClient | None = None) -> None:
    client = client or _client()
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
