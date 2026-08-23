"""紧急度评分仲裁：多先贤辩论中决定谁发言。

简历第 2 条核心公式：紧急度 = 话题相关度×w1 + 距上轮发言时长×w2 + 用户情绪/追问强度×w3
三信号全部确定性可计算，不依赖 LLM——面试可逐项拆解。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .personas import SagePersona


@dataclass
class UrgencyResult:
    """一位先贤的紧急度评分明细。"""
    sage_id: str
    relevance: float       # 话题相关度 [0, 1]
    recency: float         # 距上轮发言时长 [0, 1]（1=最久未发言）
    user_emotion: float    # 用户情绪/追问强度 [0, 1]
    total: float           # 加权总分
    rank: int              # 排名（1=first）

    # 便捷字段
    name: str = ""
    school: str = ""


# ── 权重（可调，默认值来自简历说辞） ──────────────────────────────────────

DEFAULT_WEIGHTS = {
    "relevance": 0.50,
    "recency": 0.25,
    "user_emotion": 0.25,
}


# ── 话题相关度 ────────────────────────────────────────────────────────────

def _relevance_scores(topic: str, sages: list[SagePersona],
                      embedder=None) -> dict[str, float]:
    """话题与各先贤人设的余弦相似度（BGE-M3 embedding）。

    若无 embedder 传入，回退关键词匹配（可测试、可解释）。
    """
    if embedder is not None:
        topic_vec = embedder.encode([topic])[0]
        sage_vecs = embedder.encode([s.stance_hint for s in sages])
        sims = np.dot(sage_vecs, topic_vec) / (
            np.linalg.norm(sage_vecs, axis=1) * np.linalg.norm(topic_vec) + 1e-8
        )
        # 归一化到 [0, 1]
        sims = (sims + 1) / 2
        return {s.id: float(sims[i]) for i, s in enumerate(sages)}

    # 回退：关键词匹配（确定性、可测试、面试可解释）
    return _keyword_relevance(topic, sages)


_KEYWORD_MAP = {
    # 德治/礼治/仁政 → 儒家
    "kongzi": ["德", "仁", "礼", "义", "君子", "孝", "忠", "信", "正名", "为政", "教化"],
    "mengzi": ["性善", "王道", "仁政", "民贵", "四端", "恻隐", "尧舜", "义利", "恒产", "井田"],
    "laozi": ["道", "德", "无为", "自然", "柔", "弱", "不争", "朴", "静", "玄", "天门", "谷神"],
    "hanfei": ["法", "术", "势", "刑", "赏", "罚", "耕战", "五蠹", "法治", "性恶", "变革"],
}

# 学派域→倾向先贤映射（用于辩题与学派对齐）
_SCHOOL_ALIGNMENT = {
    "德治": ["kongzi", "mengzi"],
    "法治": ["hanfei"],
    "无为": ["laozi"],
    "人治": ["kongzi", "mengzi"],
    "王道": ["mengzi", "kongzi"],
    "霸道": ["hanfei"],
    "义利": ["kongzi", "mengzi", "hanfei"],
    "人性": ["mengzi", "hanfei", "kongzi", "laozi"],
    "善恶": ["mengzi", "hanfei"],
    "礼": ["kongzi"],
    "刑": ["hanfei"],
    "自然": ["laozi"],
    "改革": ["hanfei"],
    "复古": ["kongzi"],
}


def _keyword_relevance(topic: str, sages: list[SagePersona]) -> dict[str, float]:
    """关键词匹配相关度：主题词命中先贤关键词 + 学派域对齐。"""
    scores: dict[str, float] = {}
    for s in sages:
        keywords = _KEYWORD_MAP.get(s.id, [])
        hits = sum(1 for kw in keywords if kw in topic)
        base = hits / max(len(keywords), 1) * 0.6  # 基础分

        # 学派域对齐加分
        for domain, aligned in _SCHOOL_ALIGNMENT.items():
            if domain in topic and s.id in aligned:
                base += 0.3
                break

        scores[s.id] = min(base, 1.0)
    return scores


# ── 距上轮发言时长 ────────────────────────────────────────────────────────

def _recency_scores(sage_ids: list[str],
                    last_spoken_round: dict[str, int | None],
                    current_round: int,
                    max_gap: int = 8) -> dict[str, float]:
    """距上轮发言时长 [0, 1]：越久未发言分数越高（1=最久），确保轮流发言。

    last_spoken_round: sage_id → 上轮发言的轮次（None=从未发言，给最高分）
    """
    scores: dict[str, float] = {}
    for sid in sage_ids:
        last = last_spoken_round.get(sid)
        if last is None:
            scores[sid] = 1.0  # 从未发言 → 最高优先
        else:
            gap = current_round - last
            scores[sid] = min(gap / max_gap, 1.0)
    return scores


# ── 用户情绪/追问强度 ─────────────────────────────────────────────────────

def _user_emotion_score(user_message: str) -> float:
    """用户情绪/追问强度 [0, 1]：标点密度、追问关键词、消息长度。

    纯启发式、确定性可计算——面试可逐项拆解。
    """
    if not user_message:
        return 0.5  # 中性

    # 情绪标点密度
    exclaim = user_message.count("！") + user_message.count("!")
    question = user_message.count("？") + user_message.count("?")
    punct_score = min((exclaim + question) / 8, 1.0) * 0.4

    # 追问关键词
    chase_keywords = ["请说明", "为什么", "为何", "何故", "详解", "具体", "快快", "快说",
                      "继续", "怎么", "如何", "难道", "难道不是", "莫非", "岂非",
                      "你同意", "你觉得", "你说呢", "反驳", "不认同", "不对"]
    chase_hits = sum(1 for kw in chase_keywords if kw in user_message)
    chase_score = min(chase_hits / 3, 1.0) * 0.4

    # 消息长度（长消息=更投入）
    length_score = min(len(user_message) / 200, 1.0) * 0.2

    return min(punct_score + chase_score + length_score, 1.0)


# ── 主函数 ─────────────────────────────────────────────────────────────────


def arbitrate(
    topic: str,
    sages: list[SagePersona],
    last_spoken_round: dict[str, int | None],
    current_round: int,
    user_message: str = "",
    weights: dict[str, float] | None = None,
    embedder=None,
) -> list[UrgencyResult]:
    """按紧急度评分排序，返回所有先贤的排名（index 0 = top1 发言）。"""
    w = weights or DEFAULT_WEIGHTS

    relevance = _relevance_scores(topic, sages, embedder=embedder)
    recency = _recency_scores([s.id for s in sages], last_spoken_round,
                              current_round)
    emotion = _user_emotion_score(user_message)

    results: list[UrgencyResult] = []
    for s in sages:
        rel = relevance.get(s.id, 0.0)
        rec = recency.get(s.id, 0.0)
        total = w["relevance"] * rel + w["recency"] * rec + w["user_emotion"] * emotion
        results.append(UrgencyResult(
            sage_id=s.id, relevance=rel, recency=rec,
            user_emotion=emotion, total=total, rank=0,
            name=s.name, school=s.school,
        ))

    # 降序排序
    results.sort(key=lambda r: r.total, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results


def top_speaker(results: list[UrgencyResult]) -> str:
    """返回排名第一的先贤 id。"""
    return results[0].sage_id if results else ""