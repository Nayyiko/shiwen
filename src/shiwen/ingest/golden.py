"""Golden 引据自检：验证经典原文能在 Top-K 命中目标书 + 章节前缀。"""

from __future__ import annotations

import json
from pathlib import Path

from . import milvus_store
from .embedder import get_embedder


def load_golden(path: str = "eval/golden_quotes.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(path: str = "eval/golden_quotes.json", top_k: int = 5) -> bool:
    quotes = load_golden(path)["quotes"]
    embedder = get_embedder()
    client = milvus_store._client()

    if milvus_store.count(client) == 0:
        print("[!] Milvus 为空，请先 reindex")
        return False

    passed = 0
    for q in quotes:
        vec = embedder.encode([q["text"]])[0]
        hits = milvus_store.search(vec, top_k=top_k, client=client)
        ok = any(
            h["book_id"] == q["book_id"]
            and (h["chapter"] or "").startswith(q["chapter_prefix"])
            for h in hits
        )
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {q['text'][:20]}… -> 期望 {q['book_id']}/{q['chapter_prefix']}")
        if not ok:
            top = hits[0] if hits else None
            actual = f"{top['book_id']}/{top['chapter']}" if top else "无命中"
            print(f"        实际 top1: {actual}")

    print(f"[verify] {passed}/{len(quotes)} 通过")
    return passed == len(quotes)
