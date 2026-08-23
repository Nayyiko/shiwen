"""S2 检索图集成测试（需先 reindex 入库，Docker 容器内运行）。

运行：
  docker compose run --rm backend python -m pytest tests/test_s2_retrieval.py -q
  docker compose run --rm backend python -m pytest tests/test_s2_retrieval.py -q -k "not chat_with_key"
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.shiwen.api.main import app
from src.shiwen.config import get_settings
from src.shiwen.ingest import milvus_store, golden
from src.shiwen.rag.bm25_store import search as bm25_search, clear_index
from src.shiwen.rag.fusion import rrf_fuse
from src.shiwen.rag.query_cleaner import clean_query
from src.shiwen.rag.retriever import retrieve as vector_retrieve


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def mv_client():
    """模块级 Milvus 客户端复用（避免 Milvus Lite 频繁创建/销毁连接）。"""
    return milvus_store._client()


@pytest.fixture(scope="module")
def has_data(mv_client):
    return mv_client.has_collection(milvus_store.COLLECTION) and milvus_store.count(mv_client) > 0


@pytest.fixture(scope="module")
def has_llm_key():
    return bool(get_settings().deepseek_api_key)


# ===== 查询清洗测试 =====


def test_clean_query_strips_boilerplate():
    """考据/翻译题的提问套话应被剥掉，只留名句内核。"""
    assert clean_query("吾日三省吾身是谁说的，出自哪一篇") == "吾日三省吾身"
    assert clean_query("请将论语中学而时习之，不亦说乎翻译成现代汉语") == "论语中学而时习之不亦说乎"
    assert clean_query("富贵不能淫，贫贱不能移，威武不能屈的出处") == "富贵不能淫贫贱不能移威武不能屈"
    assert clean_query("庖丁解牛的故事出自《庄子》哪一篇") == "庖丁解牛庄子"


def test_clean_query_keeps_semantic_core():
    """知识题的关键词（书名/人名/术语）应保留，不被误删。"""
    assert "道德经" in clean_query("《道德经》的作者是谁？")
    assert "孙子兵法" in clean_query("《孙子兵法》的作者是谁？")
    assert "无为而治" in clean_query("老子道德经中无为而治的思想如何理解？")
    # 清洗后为空则回退原 query
    assert clean_query("？？？") == "？？？"


# ===== BM25 检索测试 =====


def test_bm25_search(has_data):
    """BM25 应能搜到论语相关 chunk。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    hits = bm25_search("学而时习之", top_k=5)
    assert hits, "BM25 无结果"
    assert any("论语" in h["book"] for h in hits), f"BM25 应命中论语: {[h['book'] for h in hits[:3]]}"


def test_bm25_book_filter(has_data):
    """BM25 按 book_id 过滤应返回同一本书。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    hits = bm25_search("道", top_k=5, book_id="daodejing")
    assert hits
    for h in hits:
        assert h["book_id"] == "daodejing", f"过滤失效: {h['id']}"


def test_bm25_category_filter(has_data):
    """BM25 按 category 过滤应返回同一部类。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    hits = bm25_search("道", top_k=5, category="子部")
    assert hits
    for h in hits:
        assert h["category"] == "子部", f"过滤失效: {h['id']}"


# ===== RRF 融合测试 =====


def test_rrf_fusion_merges():
    """RRF 应合并两路结果并排序。"""
    # 构造两路模拟结果
    vector = [
        {"id": "a", "text": "A", "score": 0.9, "book": "论语", "chapter": "学而", "version": "通行本"},
        {"id": "b", "text": "B", "score": 0.8, "book": "孟子", "chapter": "梁惠王", "version": "通行本"},
        {"id": "c", "text": "C", "score": 0.3, "book": "道德经", "chapter": "一章", "version": "通行本"},
    ]
    bm25 = [
        {"id": "c", "text": "C", "score": 8.5, "book": "道德经", "chapter": "一章", "version": "通行本"},
        {"id": "d", "text": "D", "score": 6.0, "book": "庄子", "chapter": "逍遥游", "version": "通行本"},
    ]
    result = rrf_fuse(vector, bm25, top_k=3)
    assert len(result) == 3
    # c 在双路都出现，应排第一（双路加权）
    assert result[0]["id"] == "c", f"RRF 应把双路命中排第一，实际: {result[0]['id']}"
    assert "rrf_score" in result[0]


def test_rrf_fusion_empty():
    """空结果应返回空列表。"""
    assert rrf_fuse([], [], top_k=5) == []


# ===== 向量检索测试（保持兼容）=====


def test_vector_retrieve_golden(has_data, mv_client):
    """golden 引据问题 Top-5 命中正确书/篇。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    for q in golden.load_golden()["quotes"]:
        chunks = vector_retrieve(q["text"], top_k=5, client=mv_client)
        assert chunks, f"{q['text']} 无检索结果"
        matched = [
            c for c in chunks
            if c.book_id == q["book_id"]
            and (c.chapter or "").startswith(q["chapter_prefix"])
        ]
        assert matched, f"{q['text']} 未命中 {q['book_id']}/{q['chapter_prefix']}"


def test_vector_retrieve_metadata(has_data, mv_client):
    """每个检索结果应携带完整引据元数据。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    chunks = vector_retrieve("学而时习之", top_k=3, client=mv_client)
    assert chunks
    for c in chunks:
        assert c.book and c.chapter and c.version and c.citation, f"{c.id}: 元数据不完整"


# ===== API 端点测试 =====


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_api_search(has_data, client):
    """POST /api/search 应返回带引据的检索结果。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    resp = client.post("/api/search", json={"query": "学而时习之", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert r["book"] and r["chapter"] and r["citation"]


def test_api_chat_no_key_returns_503(client, has_llm_key):
    """未配置 DEEPSEEK_API_KEY 时 /api/chat 应返回 503。"""
    if has_llm_key:
        pytest.skip("已配置 API Key")
    resp = client.post("/api/chat", json={"query": "学而时习之"})
    assert resp.status_code == 503


def test_api_chat_with_key(has_data, has_llm_key, client):
    """端到端：/api/chat 多跳 RAG 返回带 rounds/grounding 的回答。"""
    if not has_data:
        pytest.skip("Milvus 未入库，先运行 reindex")
    if not has_llm_key:
        pytest.skip("DEEPSEEK_API_KEY 未配置")
    resp = client.post("/api/chat", json={"query": "学而时习之出自哪一篇"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"], "LLM 应返回非空回答"
    assert "学而" in data["answer"], f"回答应包含「学而」: {data['answer'][:200]}"
    assert data["citations"], "应返回引据列表"
    assert data["rounds"] >= 1, "rounds 应 ≥ 1"
    assert "grounding_pass" in data
    assert data["trace"], "应返回 trace 信息"