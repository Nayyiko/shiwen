"""S5 研究写作 单元测试（无 API key 可跑）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.shiwen.writing.citations import (
    extract_citations, verify_citations, chunk_citation_label,
)
from src.shiwen.writing.graph import _section_router
from src.shiwen.api.main import app
from src.shiwen.config import Settings


class TestCitations:
    def test_extract_single(self):
        citations = extract_citations("「论语·学而篇（通行本）」中记载了...")
        assert len(citations) == 1
        assert citations[0]["book"] == "论语"
        assert citations[0]["chapter"] == "学而篇"
        assert citations[0]["version"] == "通行本"

    def test_extract_multiple(self):
        text = "「论语·学而篇（通行本）」和「孟子·梁惠王上（通行本）」都提到..."
        citations = extract_citations(text)
        assert len(citations) == 2

    def test_extract_deduplicate(self):
        text = "「论语·学而篇（通行本）」...「论语·学而篇（通行本）」"
        citations = extract_citations(text)
        assert len(citations) == 1

    def test_extract_no_citation(self):
        assert extract_citations("这是一段没有引据的文字。") == []

    def test_verify_all_matched(self):
        chunks = [
            {"book": "论语", "chapter": "学而篇", "version": "通行本"},
            {"book": "孟子", "chapter": "梁惠王上", "version": "通行本"},
        ]
        result = verify_citations(
            "「论语·学而篇（通行本）」和「孟子·梁惠王上（通行本）」",
            chunks,
        )
        assert result["total"] == 2
        assert result["matched"] == 2
        assert result["rate"] == 1.0

    def test_verify_partial_match(self):
        chunks = [{"book": "论语", "chapter": "学而篇", "version": "通行本"}]
        result = verify_citations(
            "「论语·学而篇（通行本）」和「道德经·第一章（通行本）」",
            chunks,
        )
        assert result["total"] == 2
        assert result["matched"] == 1
        assert result["rate"] == 0.5

    def test_verify_empty(self):
        result = verify_citations("无引据", [])
        assert result["total"] == 0
        assert result["rate"] == 1.0

    def test_chunk_citation_label(self):
        c = {"book": "论语", "chapter": "学而篇", "version": "通行本"}
        assert chunk_citation_label(c) == "「论语·学而篇（通行本）」"

    def test_extract_chapter_suffix_strip(self):
        """LLM 把"通行本"写进了 chapter 名称，应剥离。"""
        # 带逗号："学而篇第一，通行本"
        citations = extract_citations("「论语·学而篇第一，通行本（通行本）」")
        assert len(citations) == 1
        assert citations[0]["book"] == "论语"
        assert citations[0]["chapter"] == "学而篇第一"
        assert citations[0]["version"] == "通行本"
        assert citations[0]["raw"] == "「论语·学而篇第一（通行本）」"

    def test_extract_chapter_suffix_no_comma(self):
        """不带逗号："颜渊篇第十二通行本" → chapter="颜渊篇第十二", version="通行本" """
        citations = extract_citations("「论语·颜渊篇第十二通行本（通行本）」")
        assert len(citations) == 1
        assert citations[0]["book"] == "论语"
        assert citations[0]["chapter"] == "颜渊篇第十二"
        assert citations[0]["version"] == "通行本"
        assert citations[0]["raw"] == "「论语·颜渊篇第十二（通行本）」"

    def test_extract_chapter_suffix_match_pool(self):
        """剥离后缀后应能匹配检索池。"""
        chunks = [{"book": "论语", "chapter": "学而篇第一", "version": "通行本"}]
        result = verify_citations(
            "「论语·学而篇第一，通行本（通行本）」",
            chunks,
        )
        assert result["total"] == 1
        assert result["matched"] == 1
        assert result["rate"] == 1.0

    def test_verify_fuzzy_prefix_match(self):
        """LLM 省略了篇名序号时，前缀匹配应生效。"""
        # 检索池有"颜渊篇第十二"，LLM 只写了"颜渊篇"
        chunks = [{"book": "论语", "chapter": "颜渊篇第十二", "version": "通行本"}]
        result = verify_citations(
            "「论语·颜渊篇（通行本）」",
            chunks,
        )
        assert result["total"] == 1
        assert result["matched"] == 1
        assert result["rate"] == 1.0

    def test_verify_fuzzy_no_false_positive(self):
        """前缀太短（<2 字符）不应误匹配。"""
        chunks = [{"book": "论语", "chapter": "学而篇第一", "version": "通行本"}]
        result = verify_citations(
            "「论语·学（通行本）」",  # 太短，不应匹配
            chunks,
        )
        assert result["total"] == 1
        assert result["matched"] == 0


class TestWritingGraph:
    def test_section_router_continue(self):
        state = {"section_index": 0, "outline": [{"title": "s1"}, {"title": "s2"}]}
        assert _section_router(state) == "retrieve"

    def test_section_router_done(self):
        state = {"section_index": 2, "outline": [{"title": "s1"}, {"title": "s2"}]}
        assert _section_router(state) == "synthesize"

    def test_section_router_empty_outline(self):
        state = {"section_index": 0, "outline": []}
        assert _section_router(state) == "synthesize"

    def test_resume_skip_done(self):
        """断点续写：section_index 应从已完成处继续。"""
        state = {"section_index": 1, "outline": [
            {"title": "s1", "text": "done", "done": True},
            {"title": "s2", "text": "", "done": False},
        ]}
        assert _section_router(state) == "retrieve"


class TestWritingAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_503_without_key(self, client):
        with patch('src.shiwen.api.main.get_settings') as m:
            m.return_value = Settings(deepseek_api_key="")
            r = client.post("/api/write", json={"topic": "论语中的仁学思想"})
            assert r.status_code == 503

    def test_200_mock_graph(self, client):
        with patch('src.shiwen.api.main.get_settings') as ms, \
             patch('src.shiwen.writing.graph.build_writing_graph') as mb:
            ms.return_value = Settings(deepseek_api_key="test-key")
            g = MagicMock()
            g.invoke.return_value = {
                "outline": [
                    {"title": "仁的定义", "text": "正文...", "chunks": [
                        {"book": "论语", "chapter": "学而篇", "version": "通行本", "text": "..."}
                    ]},
                ],
                "all_chunks": [
                    {"book": "论语", "chapter": "学而篇", "version": "通行本", "text": "..."}
                ],
                "article": "> 引言...\n\n## 仁的定义\n正文...\n\n## 引据清单\n- 论语",
                "trace": [{"node": "outline", "elapsed_ms": 0}],
            }
            mb.return_value = g
            r = client.post("/api/write", json={"topic": "论语中的仁学思想"})
            assert r.status_code == 200
            d = r.json()
            assert d["topic"] == "论语中的仁学思想"
            assert len(d["sections"]) == 1
            assert d["sections"][0]["title"] == "仁的定义"
            assert len(d["citations"]) == 1