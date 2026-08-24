"""上下文治理 + 分层记忆 单元测试（无 Redis 可跑，走降级模式）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shiwen.context import (
    estimate_tokens,
    trim_history,
    compress_chunks,
    build_prompt_budget,
)
from src.shiwen.memory import MemoryManager
from src.shiwen.redis_store import RedisStore


def _memory() -> MemoryManager:
    """降级模式 MemoryManager（进程内 dict，无 Redis）。"""
    return MemoryManager(store=RedisStore(enabled=False))


class TestContext:
    def test_estimate_tokens(self):
        assert estimate_tokens("") == 0
        # 中文约 1 字符 1 token
        assert estimate_tokens("学而时习之") == 5
        # 英文约 4 字符 1 token
        assert estimate_tokens("abcd") == 1

    def test_trim_history_keeps_recent(self):
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "善哉"},
            {"role": "user", "content": "何为仁？"},
            {"role": "assistant", "content": "仁者爱人"},
        ]
        # 预算足够 → 全保留
        kept = trim_history(history, max_tokens=100)
        assert len(kept) == 4
        # 预算极小 → 只保留最新一条
        kept = trim_history(history, max_tokens=1)
        assert len(kept) == 1
        assert kept[0]["content"] == "仁者爱人"

    def test_trim_history_empty(self):
        assert trim_history([]) == []

    def test_compress_chunks_truncates(self):
        chunks = [
            {"text": "A" * 2000, "book": "论语", "chapter": "学而篇第一"},
        ]
        out = compress_chunks(chunks, chunk_max_chars=600)
        assert len(out[0]["text"]) <= 600 + 1  # 截断 + "…"

    def test_compress_chunks_budget(self):
        chunks = [
            {"text": "内容" * 100},  # 200 字符
            {"text": "内容" * 100},
            {"text": "内容" * 100},
        ]
        # 预算只够 2 条（每条 200 token + 20 元数据开销）
        out = compress_chunks(chunks, max_tokens=240)
        assert len(out) == 1

    def test_build_prompt_budget(self):
        history = [{"role": "user", "content": "你好"}] * 10
        chunks = [{"text": "内容" * 100}] * 5
        result = build_prompt_budget(history, chunks)
        assert "history" in result
        assert "chunks" in result
        assert result["total_tokens"] == result["history_tokens"] + result["chunk_tokens"]


class TestMemory:
    def test_short_term_memory(self):
        m = _memory()
        m.append_message("s1", "user", "何为仁？")
        m.append_message("s1", "assistant", "仁者爱人")
        msgs = m.get_recent_messages("s1", n=6)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["content"] == "仁者爱人"

    def test_short_term_memory_isolated_by_session(self):
        m = _memory()
        m.append_message("s1", "user", "A")
        m.append_message("s2", "user", "B")
        assert len(m.get_recent_messages("s1")) == 1
        assert m.get_recent_messages("s1")[0]["content"] == "A"
        assert m.get_recent_messages("s2")[0]["content"] == "B"

    def test_long_term_persona(self):
        m = _memory()
        p = m.get_persona("kongzi")
        assert p is not None
        assert p.name == "孔子"
        assert m.get_persona("not_exist") is None

    def test_task_state_roundtrip(self):
        m = _memory()
        state = {"section_index": 2, "outline": [{"title": "s1"}, {"title": "s2"}]}
        m.save_task_state("w1", state)
        loaded = m.load_task_state("w1")
        assert loaded == state

    def test_task_state_not_found(self):
        m = _memory()
        assert m.load_task_state("no_such_task") is None

    def test_task_state_clear(self):
        m = _memory()
        m.save_task_state("t1", {"x": 1})
        m.clear_task_state("t1")
        assert m.load_task_state("t1") is None