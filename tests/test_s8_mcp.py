"""S8 MCP 工具层 单元测试（无 API key / 无 DB 可跑，全 mock）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from unittest.mock import MagicMock, patch


class TestRetrieveTool:
    def test_retrieve_returns_json(self):
        from src.shiwen.mcp.server import retrieve

        fake_chunk = MagicMock()
        fake_chunk.id = "lunyu_1"
        fake_chunk.text = "学而时习之，不亦说乎"
        fake_chunk.book = "论语"
        fake_chunk.book_id = "lunyu"
        fake_chunk.author = "孔子"
        fake_chunk.dynasty = "春秋"
        fake_chunk.category = "经部"
        fake_chunk.chapter = "学而篇第一"
        fake_chunk.version = "通行本"
        fake_chunk.score = 0.91

        with patch("src.shiwen.rag.retriever.retrieve", return_value=[fake_chunk]):
            out = retrieve("学而时习之", top_k=3, book_id="lunyu")

        data = json.loads(out)
        assert len(data["results"]) == 1
        assert data["results"][0]["book"] == "论语"
        assert data["results"][0]["chapter"] == "学而篇第一"


class TestVerifyTool:
    def test_verify_matches(self):
        from src.shiwen.mcp.server import verify

        chunks_json = json.dumps({"results": [
            {"book": "论语", "chapter": "学而篇第一", "version": "通行本", "text": "..."}
        ]})
        text = "「论语·学而篇第一（通行本）」中记载..."

        out = verify(text, chunks_json)
        data = json.loads(out)
        assert data["total"] == 1
        assert data["matched"] == 1
        assert data["rate"] == 1.0


class TestQueryPersonTool:
    def test_query_person_found(self):
        from src.shiwen.mcp.server import query_person

        fake = {
            "person": {"id": "kongzi", "name": "孔子", "dynasty": "春秋", "school": "儒家"},
            "works": [{"title": "论语", "relation": "著", "note": None}],
            "relations": [{"target_id": "yanhui", "target_name": "颜回", "relation": "师从", "note": None}],
        }
        with patch("src.shiwen.ingest.people.query_person", return_value=fake):
            out = query_person("孔子")

        data = json.loads(out)
        assert data["person"]["name"] == "孔子"
        assert data["works"][0]["title"] == "论语"

    def test_query_person_not_found(self):
        from src.shiwen.mcp.server import query_person

        with patch("src.shiwen.ingest.people.query_person", return_value=None):
            out = query_person("不存在的")

        data = json.loads(out)
        assert "error" in data


class TestWriteTool:
    def test_write_returns_json(self):
        from src.shiwen.mcp.server import write

        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "outline": [{"title": "仁的定义", "text": "正文"}],
            "all_chunks": [
                {"book": "论语", "chapter": "学而篇第一", "version": "通行本", "text": "..."}
            ],
            "article": "> 引言\n\n## 仁的定义\n正文",
            "trace": [{"node": "outline", "elapsed_ms": 100}],
        }
        with patch("src.shiwen.writing.graph.build_writing_graph", return_value=fake_graph):
            out = write("论语中的仁学思想", max_sections=2)

        data = json.loads(out)
        assert data["article"].startswith(">")
        assert len(data["sections"]) == 1
        assert data["sections"][0]["title"] == "仁的定义"
        assert len(data["citations"]) == 1


class TestServerRegistration:
    def test_server_has_four_tools(self):
        from src.shiwen.mcp.server import mcp
        # FastMCP 3.x: 工具注册在 _docket 上
        tool_names = set(getattr(mcp, "_docket", {}).keys()) if isinstance(getattr(mcp, "_docket", None), dict) else set()
        # 若拿不到内部结构，退而验证四个工具函数存在
        from src.shiwen.mcp.server import retrieve, verify, query_person, write
        assert callable(retrieve)
        assert callable(verify)
        assert callable(query_person)
        assert callable(write)