"""S6 新裁角色扮演 单元测试（无 API key 可跑）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.shiwen.roleplay.graph import build_roleplay_graph, RoleplayState
from src.shiwen.api.main import app
from src.shiwen.config import Settings


class TestRoleplayGraph:
    """角色扮演图结构测试。"""

    def test_graph_structure(self):
        """图编译成功，retrieve → generate → END。"""
        graph = build_roleplay_graph()
        assert graph is not None
        # 验证节点存在
        nodes = graph.get_graph().nodes
        assert "retrieve" in nodes
        assert "generate" in nodes

    def test_state_shape(self):
        """验证状态字段完整性。"""
        state: RoleplayState = {
            "sage_id": "kongzi",
            "user_message": "敢问夫子，何为仁？",
            "history": [],
            "chunks": [],
            "response": "",
            "trace": [],
        }
        assert state["sage_id"] == "kongzi"
        assert "user_message" in state
        assert "history" in state
        assert "response" in state

    def test_unknown_sage_returns_graceful(self):
        """未知先贤 ID 应返回友好提示。"""
        graph = build_roleplay_graph()
        result = graph.invoke({
            "sage_id": "sunwukong",
            "user_message": "你好",
            "history": [],
            "chunks": [],
            "response": "",
            "trace": [],
        })
        assert "尚未就座" in result["response"]


class TestRoleplayAPI:
    """API 端点测试。"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_503_without_key(self, client):
        with patch('src.shiwen.api.main.get_settings') as m:
            m.return_value = Settings(deepseek_api_key="")
            r = client.post("/api/roleplay", json={
                "sage_id": "kongzi",
                "message": "何为仁？",
            })
            assert r.status_code == 503

    def test_400_unknown_sage(self, client):
        with patch('src.shiwen.api.main.get_settings') as ms:
            ms.return_value = Settings(deepseek_api_key="test-key")
            r = client.post("/api/roleplay", json={
                "sage_id": "unknown",
                "message": "你好",
            })
            assert r.status_code == 400

    def test_200_mock_graph(self, client):
        with patch('src.shiwen.api.main.get_settings') as ms, \
             patch('src.shiwen.roleplay.graph.build_roleplay_graph') as mb:
            ms.return_value = Settings(deepseek_api_key="test-key")
            g = MagicMock()
            g.invoke.return_value = {
                "sage_id": "kongzi",
                "user_message": "何为仁？",
                "history": [],
                "chunks": [
                    {"book": "论语", "chapter": "颜渊篇第十二", "version": "通行本",
                     "text": "樊迟问仁。子曰：爱人。"},
                ],
                "response": "仁者，爱人也。克己复礼为仁。",
                "trace": [{"node": "retrieve", "elapsed_ms": 50},
                          {"node": "generate", "elapsed_ms": 800}],
            }
            mb.return_value = g
            r = client.post("/api/roleplay", json={
                "sage_id": "kongzi",
                "message": "何为仁？",
            })
            assert r.status_code == 200
            d = r.json()
            assert d["sage_id"] == "kongzi"
            assert d["sage_name"] == "孔子"
            assert d["school"] == "儒家"
            assert "爱人" in d["response"]
            assert len(d["citations"]) == 1
            assert d["citations"][0]["book"] == "论语"

    def test_multi_turn_with_history(self, client):
        """多轮对话：history 正确传递。"""
        with patch('src.shiwen.api.main.get_settings') as ms, \
             patch('src.shiwen.roleplay.graph.build_roleplay_graph') as mb:
            ms.return_value = Settings(deepseek_api_key="test-key")
            g = MagicMock()
            g.invoke.return_value = {
                "sage_id": "kongzi",
                "response": "善哉此问。",
                "chunks": [],
                "trace": [],
            }
            mb.return_value = g
            r = client.post("/api/roleplay", json={
                "sage_id": "kongzi",
                "message": "何以守仁？",
                "history": [
                    {"role": "user", "content": "何为仁？"},
                    {"role": "assistant", "content": "仁者，爱人也。"},
                ],
            })
            assert r.status_code == 200
            # 验证 history 传入了 graph
            call_args = g.invoke.call_args[0][0]
            assert len(call_args["history"]) == 2


class TestRoleplayCitations:
    """角色扮演引据验证：复用 S5 citations 模块。"""

    def test_roleplay_citation_traceable(self):
        from src.shiwen.writing.citations import verify_citations

        response = "子曰：「克己复礼为仁。」（论语·颜渊篇第十二（通行本））"
        chunks = [
            {"book": "论语", "chapter": "颜渊篇第十二", "version": "通行本", "text": "..."},
        ]
        result = verify_citations(response, chunks)
        assert result["total"] >= 1
        assert result["matched"] >= 1

    def test_roleplay_no_fabricated_citation(self):
        from src.shiwen.writing.citations import verify_citations

        response = "据《论语·不存在的篇章（通行本）》记载..."
        chunks = [
            {"book": "论语", "chapter": "学而篇第一", "version": "通行本", "text": "..."},
        ]
        result = verify_citations(response, chunks)
        # 编造的引据不应命中
        unmatched = [u["chapter"] for u in result["unmatched"]]
        assert "不存在的篇章" in unmatched