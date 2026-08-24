"""Redis 客户端封装：会话状态 / 分层记忆 / 断点恢复的存储层。

懒加载单例。redis_enabled=False 或连接失败时降级为进程内 dict（本地无 Redis 可跑、单测可用）。
分层记忆统一以 JSON 字符串存储，key 带前缀区分 namespace。
"""

from __future__ import annotations

import json
from functools import lru_cache

from src.shiwen.config import get_settings


class RedisStore:
    """Redis 封装，带进程内降级。"""

    def __init__(self, host: str | None = None, port: int | None = None,
                 db: int | None = None, enabled: bool | None = None):
        s = get_settings()
        self.host = host or s.redis_host
        self.port = port or s.redis_port
        self.db = db if db is not None else s.redis_db
        self.enabled = enabled if enabled is not None else s.redis_enabled
        self._client = None
        self._fallback: dict[str, str] = {}  # 进程内降级

    def _load(self):
        """懒加载连接；失败返回 None（走降级）。"""
        if self._client is None and self.enabled:
            try:
                import redis
                client = redis.Redis(
                    host=self.host, port=self.port, db=self.db, decode_responses=True)
                client.ping()
                self._client = client
            except Exception:
                self._client = None
        return self._client

    # ── 基础 KV（字符串） ────────────────────────────────────────────────

    def set(self, key: str, value: str, ttl: int | None = None) -> None:
        c = self._load()
        if c:
            c.set(key, value, ex=ttl)
        else:
            self._fallback[key] = value

    def get(self, key: str) -> str | None:
        c = self._load()
        if c:
            return c.get(key)
        return self._fallback.get(key)

    def delete(self, key: str) -> None:
        c = self._load()
        if c:
            c.delete(key)
        self._fallback.pop(key, None)

    def exists(self, key: str) -> bool:
        c = self._load()
        if c:
            return bool(c.exists(key))
        return key in self._fallback

    # ── JSON 便捷方法 ──────────────────────────────────────────────────

    def set_json(self, key: str, obj, ttl: int | None = None) -> None:
        self.set(key, json.dumps(obj, ensure_ascii=False), ttl)

    def get_json(self, key: str):
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    # ── 列表（会话历史） ────────────────────────────────────────────────

    def lpush(self, key: str, value: str) -> None:
        c = self._load()
        if c:
            c.lpush(key, value)
        else:
            self._fallback.setdefault(key, "[]")
            # fallback 用 JSON 数组存，历史新条目插头部
            try:
                lst = json.loads(self._fallback[key])
            except (json.JSONDecodeError, TypeError):
                lst = []
            lst.insert(0, value)
            self._fallback[key] = json.dumps(lst, ensure_ascii=False)

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        c = self._load()
        if c:
            return c.lrange(key, start, end)
        try:
            lst = json.loads(self._fallback.get(key, "[]"))
        except (json.JSONDecodeError, TypeError):
            return []
        if end < 0:
            return lst[start:]
        return lst[start:end + 1]

    def ltrim(self, key: str, start: int, end: int) -> None:
        c = self._load()
        if c:
            c.ltrim(key, start, end)
        else:
            try:
                lst = json.loads(self._fallback.get(key, "[]"))
            except (json.JSONDecodeError, TypeError):
                lst = []
            self._fallback[key] = json.dumps(lst[start:end + 1], ensure_ascii=False)


@lru_cache
def get_redis() -> RedisStore:
    return RedisStore()
