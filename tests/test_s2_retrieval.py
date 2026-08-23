"""S2 检索层集成测试（需先 reindex 入库，Docker 容器内运行）。

运行：docker compose run --rm backend python -m pytest tests/test_s2_retrieval.py -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.shiwen.api.main import app
from src.shiwen.config import get_settings
from src.shiwen.ingest import milvus_store, golden
from src.shiwen.rag.retriever import retrieve


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def has_data():
    mv = milvus_store._client()
    return mv.has_collection(milvus_store.COLLECTION) and milvus_store.count(mv) > 0


@pytest.fixture(scope="module")
def has_llm_key():
    return bool(get_settings().deepseek_api_key)


# ===== 检索层测试（不需要 LLM Key）=====


def test_retrieve_golden_hits_target(has_data):
    """golden 引据问题应在 Top-5 中命中正确书/篇。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    for q in golden.load_golden()["quotes"]:
        chunks = retrieve(q["text"], top_k=5)
        assert chunks, f"{q['text']} 无检索结果"
        matched = [
            c for c in chunks
            if c.book_id == q["book_id"]
            and (c.chapter or "").startswith(q["chapter_prefix"])
        ]
        assert matched, (
            f"{q['text']} 未命中 {q['book_id']}/{q['chapter_prefix']}，"
            f"top1={chunks[0].book_id}/{chunks[0].chapter}"
        )


def test_retrieve_returns_citation_metadata(has_data):
    """每个检索结果应携带完整的引据元数据。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    chunks = retrieve("学而时习之", top_k=3)
    assert chunks
    for c in chunks:
        assert c.book, f"{c.id}: book 为空"
        assert c.chapter, f"{c.id}: chapter 为空"
        assert c.version, f"{c.id}: version 为空"
        assert c.citation, f"{c.id}: citation 为空"


def test_retrieve_book_filter(has_data):
    """按 book_id 过滤：所有结果应是同一本书。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    chunks = retrieve("道", top_k=5, book_id="lunyu")
    assert chunks
    for c in chunks:
        assert c.book_id == "lunyu", f"过滤失效：{c.id} 是 {c.book_id}"


def test_retrieve_category_filter(has_data):
    """按 category 过滤：所有结果应是同一部类。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    chunks = retrieve("道", top_k=5, category="子部")
    assert chunks
    for c in chunks:
        assert c.category == "子部", f"过滤失效：{c.id} 是 {c.category}"


# ===== API 端点测试 =====


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_search(has_data, client):
    """POST /api/search 应返回带引据的检索结果。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    resp = client.post("/api/search", json={"query": "学而时习之", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "学而时习之"
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert r["book"]
        assert r["chapter"]
        assert r["citation"]


def test_api_chat_no_key_returns_503(client, has_llm_key):
    """未配置 DEEPSEEK_API_KEY 时 /api/chat 应返回 503 并提示。"""
    if has_llm_key:
        pytest.skip("已配置 API Key，跳过无 key 测试")
    resp = client.post("/api/chat", json={"query": "学而时习之"})
    assert resp.status_code == 503
    assert "DEEPSEEK_API_KEY" in resp.json()["detail"]


def test_api_chat_with_key(has_data, has_llm_key, client):
    """端到端：/api/chat 检索 + 生成，回答应包含引据。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    if not has_llm_key:
        pytest.skip("DEEPSEEK_API_KEY 未配置")
    resp = client.post("/api/chat", json={"query": "学而时习之出自哪一篇"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"], "LLM 应返回非空回答"
    assert "学而" in data["answer"], f"回答应包含「学而」：{data['answer'][:200]}"
    assert data["citations"], "应返回引据列表"