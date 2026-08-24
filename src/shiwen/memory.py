"""分层记忆管理器：短期对话 / 长期人设 / 任务状态。

简历第 3 条「Agent 状态与上下文治理」的落地：
- 短期对话记忆：会话内近期消息，Redis list（带 TTL），滚动窗口保留最近 N 条
- 长期人设记忆：先贤人格（personas.py 的 SagePersona + PG person 表），跨会话稳定
- 任务状态：进行中的任务（写作 section_index / 辩论 round），Redis JSON 持久化，支持断点恢复

存储层走 RedisStore，redis_enabled=False 时自动降级为进程内（本地/单测可跑）。
"""

from __future__ import annotations

import json

from src.shiwen.agents.personas import SAGES, SagePersona
from src.shiwen.redis_store import RedisStore, get_redis

# 短期记忆滚动窗口上限（防长对话无限膨胀）
MAX_HISTORY = 20
# 短期记忆默认 TTL（秒）：2 小时无活动自动过期
DEFAULT_TTL = 2 * 60 * 60


class MemoryManager:
    """三层记忆统一入口。"""

    def __init__(self, store: RedisStore | None = None):
        self.store = store or get_redis()

    # ── 短期对话记忆 ────────────────────────────────────────────────────

    def _history_key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """追加一条对话消息（头部插入，保留最近 MAX_HISTORY 条）。"""
        key = self._history_key(session_id)
        self.store.lpush(key, json.dumps({"role": role, "content": content},
                                         ensure_ascii=False))
        self.store.ltrim(key, 0, MAX_HISTORY - 1)

    def get_recent_messages(self, session_id: str, n: int = 6) -> list[dict]:
        """获取最近 n 条对话（正序返回，供 prompt 拼接）。"""
        key = self._history_key(session_id)
        msgs = self.store.lrange(key, 0, n - 1)  # 最新在前
        msgs.reverse()  # 翻转为正序
        out: list[dict] = []
        for m in msgs:
            try:
                out.append(json.loads(m))
            except (json.JSONDecodeError, TypeError):
                continue
        return out

    # ── 长期人设记忆 ────────────────────────────────────────────────────

    def get_persona(self, sage_id: str) -> SagePersona | None:
        """获取先贤长期人设（跨会话稳定的学派立场/语言风格/必引己著）。"""
        return SAGES.get(sage_id)

    # ── 任务状态（断点恢复） ────────────────────────────────────────────

    def _task_key(self, task_id: str) -> str:
        return f"task:{task_id}"

    def save_task_state(self, task_id: str, state: dict, ttl: int = DEFAULT_TTL) -> None:
        """持久化任务状态（如写作 section_index/outline、辩论 round），支持断点恢复。"""
        self.store.set_json(self._task_key(task_id), state, ttl)

    def load_task_state(self, task_id: str) -> dict | None:
        """加载任务状态；不存在返回 None。"""
        return self.store.get_json(self._task_key(task_id))

    def clear_task_state(self, task_id: str) -> None:
        self.store.delete(self._task_key(task_id))
