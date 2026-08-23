"""Golden 引据集成测试（需先 reindex 入库，Docker 容器内运行）。

运行：docker compose run --rm backend python -m pytest -q
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from src.shiwen.ingest import golden, milvus_store, pg_store
from src.shiwen.ingest.embedder import get_embedder
from src.shiwen.ingest.pg_store import ChunkRow


@pytest.fixture(scope="module")
def client():
    return milvus_store._client()


@pytest.fixture(scope="module")
def has_data(client) -> bool:
    return client.has_collection(milvus_store.COLLECTION) and milvus_store.count(client) > 0


def test_golden_quotes_hit_target(client, has_data):
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    embedder = get_embedder()
    for q in golden.load_golden()["quotes"]:
        vec = embedder.encode([q["text"]])[0]
        hits = milvus_store.search(vec, top_k=5, client=client)
        assert hits, f"{q['text']} 无命中"
        matched = [
            h for h in hits
            if h["book_id"] == q["book_id"]
            and (h["chapter"] or "").startswith(q["chapter_prefix"])
        ]
        assert matched, (
            f"{q['text']} 未命中 {q['book_id']}/{q['chapter_prefix']}，"
            f"top1={hits[0]['book_id']}/{hits[0]['chapter']}"
        )


def test_pg_id_consistent(client, has_data):
    if not has_data:
        pytest.skip("PG 未入库，先运行 reindex")
    rows = milvus_store.query('id != ""', limit=1, client=client)
    assert rows, "Milvus 无记录"
    cid = rows[0]["id"]

    engine = pg_store.get_engine()
    with engine.connect() as conn:
        # 用 select(ChunkRow.text) 显式取列：Connection.execute 不重建 ORM 实体，
        # select(ChunkRow) 会展开为逐列原始值，scalar_one_or_none 返回首列 id(str)
        row_text = conn.execute(
            select(ChunkRow.text).where(ChunkRow.id == cid)
        ).scalar_one_or_none()
    assert row_text is not None, f"PG 中找不到同 id 记录 {cid}"
    assert row_text == rows[0]["text"], "Milvus 与 PG 同 id 的 text 不一致"
