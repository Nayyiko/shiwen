"""上下文治理：token 预算跟踪 + 检索上下文压缩 + 动态裁剪上下文窗口。

简历第 3 条「Agent 状态与上下文治理」的落地，控制长对话成本：
- token 预算：估算 prompt 各部分 token，超预算时逐级裁剪
- 检索上下文压缩：chunk 超预算时按优先级截断（先截单条文本、再减 chunk 数）
- 动态裁剪窗口：对话历史只保留"最近 + 足够 token 预算"的窗口

确定性纯函数，无副作用，可单测。
"""

from __future__ import annotations

# 默认 token 预算（中文为主，约 1 字符 ≈ 1 token，此处取保守系数）
HISTORY_TOKEN_BUDGET = 2000   # 对话历史预算
CHUNK_TOKEN_BUDGET = 3000     # 检索资料预算
CHUNK_MAX_CHARS = 600         # 单条 chunk 截断上限


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。中文约 1 字符 ≈ 1 token，英文约 4 字符 ≈ 1 token。

    无精确 tokenizer（tiktoken 对中文不准），用保守字符估算，宁多勿少。
    """
    if not text:
        return 0
    # 粗略：中文按 1 字符 1 token，ASCII 按 4 字符 1 token
    cjk = sum(1 for ch in text if '一' <= ch <= '鿿')
    ascii_chars = len(text) - cjk
    return cjk + (ascii_chars + 3) // 4


def trim_history(history: list[dict], max_tokens: int = HISTORY_TOKEN_BUDGET) -> list[dict]:
    """动态裁剪对话历史：从最新往回保留，直到 token 预算用尽。

    Args:
        history: [{role, content}, ...] 正序
        max_tokens: token 预算上限

    Returns:
        裁剪后的历史（正序，保留最近的）
    """
    if not history:
        return []

    kept: list[dict] = []
    used = 0
    for h in reversed(history):  # 从最新往回
        cost = estimate_tokens(h.get("content", ""))
        if used + cost > max_tokens and kept:
            break  # 预算用尽，停止（至少保留一条最新）
        kept.append(h)
        used += cost

    kept.reverse()  # 恢复正序
    return kept


def compress_chunks(chunks: list[dict], max_tokens: int = CHUNK_TOKEN_BUDGET,
                    chunk_max_chars: int = CHUNK_MAX_CHARS) -> list[dict]:
    """压缩检索上下文：先截断每条 chunk 的 text，再超预算则减少 chunk 数。

    返回新的 chunk 列表（不修改原对象），每条含 text 截断后的副本。
    """
    if not chunks:
        return []

    # 第一级：截断单条 chunk 文本
    trimmed: list[dict] = []
    for c in chunks:
        text = c.get("text", "")
        if len(text) > chunk_max_chars:
            c2 = dict(c)
            c2["text"] = text[:chunk_max_chars] + "…"
            trimmed.append(c2)
        else:
            trimmed.append(dict(c))

    # 第二级：按 token 预算裁剪 chunk 数量（保留靠前的，检索已按相关度排序）
    kept: list[dict] = []
    used = 0
    for c in trimmed:
        cost = estimate_tokens(c.get("text", "")) + 20  # 元数据开销
        if used + cost > max_tokens and kept:
            break
        kept.append(c)
        used += cost

    return kept


def build_prompt_budget(history: list[dict], chunks: list[dict],
                        history_budget: int = HISTORY_TOKEN_BUDGET,
                        chunk_budget: int = CHUNK_TOKEN_BUDGET) -> dict:
    """一次性对历史 + 检索资料做 token 预算裁剪，返回裁剪后的两部分 + 预算统计。

    供各图生成节点调用，统一控制 prompt 体积。
    """
    trimmed_history = trim_history(history, history_budget)
    trimmed_chunks = compress_chunks(chunks, chunk_budget)
    history_tokens = sum(estimate_tokens(h.get("content", "")) for h in trimmed_history)
    chunk_tokens = sum(estimate_tokens(c.get("text", "")) for c in trimmed_chunks)
    return {
        "history": trimmed_history,
        "chunks": trimmed_chunks,
        "history_tokens": history_tokens,
        "chunk_tokens": chunk_tokens,
        "total_tokens": history_tokens + chunk_tokens,
    }
