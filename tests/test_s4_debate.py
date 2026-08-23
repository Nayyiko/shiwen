"""S4 先贤辩论 单元测试（无 API key 可跑）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.shiwen.agents.personas import SAGES, KONGZI, MENGZI, LAOZI, HANFEI
from src.shiwen.agents.urgency import (
    arbitrate, _keyword_relevance, _recency_scores, _user_emotion_score, top_speaker,
)
from src.shiwen.agents.drift import DriftMonitor
from src.shiwen.agents.debate import _topic_parse, _route_after_speak
from src.shiwen.api.main import app
from src.shiwen.config import Settings


class TestUrgency:
    def test_keyword_relevance_de_rule(self):
        scores = _keyword_relevance("德治与法治哪个更适合治国",
                                    [KONGZI, MENGZI, LAOZI, HANFEI])
        assert scores["hanfei"] > 0.3
        assert scores["kongzi"] > 0.3

    def test_keyword_relevance_wuwei(self):
        scores = _keyword_relevance("无为而治是否可行",
                                    [KONGZI, MENGZI, LAOZI, HANFEI])
        assert scores["laozi"] == max(scores.values())

    def test_recency_never_spoken_max(self):
        scores = _recency_scores(
            ["kongzi", "mengzi", "laozi", "hanfei"],
            {"kongzi": 0, "mengzi": 1}, current_round=2)
        assert scores["laozi"] == 1.0
        assert scores["hanfei"] == 1.0

    def test_recency_monotonic(self):
        scores = _recency_scores(
            ["kongzi", "mengzi"], {"kongzi": 0, "mengzi": 5}, current_round=10)
        assert scores["kongzi"] > scores["mengzi"]

    def test_user_emotion_high_intensity(self):
        high = _user_emotion_score("为什么！快说！你难道不认同吗？？")
        low = _user_emotion_score("好的")
        assert high > low and high > 0.5

    def test_arbitrate_de_rule_topic(self):
        results = arbitrate(
            topic="德治与法治哪个更适合治国",
            sages=[KONGZI, MENGZI, LAOZI, HANFEI],
            last_spoken_round={}, current_round=0)
        top2 = {r.sage_id for r in results[:2]}
        assert top2 & {"kongzi", "hanfei"}

    def test_top_speaker(self):
        results = arbitrate(
            topic="法治与人治",
            sages=[KONGZI, MENGZI, LAOZI, HANFEI],
            last_spoken_round={}, current_round=0)
        assert top_speaker(results) in {"kongzi", "hanfei"}


class TestDrift:
    def test_no_drift_same_style(self):
        monitor = DriftMonitor(window_size=3, threshold=0.5)

        class Emb:
            def encode(self, texts):
                return np.array([[1.0, 0.0, 0.0]] * len(texts))

        for _ in range(4):
            event = monitor.observe("kongzi", "test text", Emb())
        assert event is None

    def test_drift_detected_style_change(self):
        monitor = DriftMonitor(window_size=3, threshold=0.5)

        class Emb:
            def encode(self, texts):
                vecs = []
                for t in texts:
                    if "法家" in t:
                        vecs.append([0.0, 1.0, 0.0])
                    else:
                        vecs.append([1.0, 0.0, 0.0])
                return np.array(vecs)

        emb = Emb()
        for _ in range(3):
            monitor.observe("kongzi", "子曰：为政以德", emb)
        event = monitor.observe("kongzi", "法家主张以法治国", emb)
        assert event is not None
        assert event.similarity < 0.5

    def test_correction_hint_format(self):
        monitor = DriftMonitor()
        hint = monitor.get_correction_hint(
            "kongzi", "孔子", ["子曰：学而时习之", "子曰：为政以德"])
        assert "学而时习之" in hint
        assert "风格提醒" in hint


class TestDebateGraphNodes:
    def test_topic_parse_initializes(self):
        result = _topic_parse({"topic": "德治与法治", "max_speeches": 4})
        assert result["round"] == 0
        assert result["speech_log"] == []
        assert result["summary"] == ""

    def test_route_continue(self):
        assert _route_after_speak({"round": 3, "max_speeches": 8}) == "arbitrate"

    def test_route_summarize(self):
        assert _route_after_speak({"round": 8, "max_speeches": 8}) == "summarize"


class TestDebateAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_503_without_key(self, client):
        with patch('src.shiwen.api.main.get_settings') as m:
            m.return_value = Settings(deepseek_api_key="")
            r = client.post("/api/debate", json={"topic": "test"})
            assert r.status_code == 503

    def test_200_mock_graph(self, client):
        with patch('src.shiwen.api.main.get_settings') as ms, \
             patch('src.shiwen.agents.debate.build_debate_graph') as mb:
            ms.return_value = Settings(deepseek_api_key="test-key")
            g = MagicMock()
            g.invoke.return_value = {
                "speech_log": [
                    {"sage_id": "kongzi", "name": "孔子", "school": "儒家",
                     "text": "子曰...", "citations": [
                         {"book": "论语", "chapter": "为政篇", "version": "通行本", "text": "..."}
                     ], "urgency_rank": 1},
                ],
                "summary": "总结",
                "urgency_trace": [], "drift_events": [],
                "trace": [{"node": "t", "elapsed_ms": 0}],
            }
            mb.return_value = g
            r = client.post("/api/debate", json={"topic": "德治与法治"})
            assert r.status_code == 200
            d = r.json()
            assert d["summary"] == "总结"
            assert len(d["speeches"]) == 1
            assert d["speeches"][0]["sage_id"] == "kongzi"


class TestRouting:
    @pytest.mark.parametrize("topic,expected_top2", [
        ("德治与法治哪个更适合治国", {"kongzi", "hanfei"}),
        ("人性本善还是本恶", {"mengzi", "hanfei"}),
        ("义利之辨：君子应该重义还是重利", {"kongzi", "mengzi"}),
        ("无为而治是否可行", {"laozi", "kongzi"}),
        ("王道与霸道孰优孰劣", {"mengzi", "hanfei"}),
    ])
    def test_routing(self, topic, expected_top2):
        results = arbitrate(
            topic=topic, sages=[KONGZI, MENGZI, LAOZI, HANFEI],
            last_spoken_round={}, current_round=0)
        top2 = {r.sage_id for r in results[:2]}
        assert top2 & expected_top2, \
            f"辩题「{topic}」期望 top2 包含 {expected_top2}，实际 {top2}"