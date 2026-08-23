"""Grounding 校验 + 反思诊断：LLM-as-judge，简历"引据合规率"的在线实现。

- grounding 校验：判回答是否被检索 chunk 支撑，同时校验引据合规（书·篇·版本一致性）
- 反思诊断：分析 why miss（实体抽取错 / Query 过宽 / 语料缺失），产出改写 query
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from src.shiwen.config import get_settings


@dataclass
class GroundingResult:
    passed: bool
    reason: str
    unsupported_claims: list[str] = field(default_factory=list)


_GROUNDING_PROMPT = """你是古籍引据合规校验员。请严格审查以下回答：

## 回答
{answer}

## 检索资料（回答只能基于这些资料）
{chunks_text}

## 任务
1. 逐条检查回答中的**事实主张**（尤其是引用原文、标注出处的主张），是否能在检索资料中找到原文支撑。
2. 检查引据标注（书·篇·版本）是否与检索资料的元数据一致。版本不对、篇名不对、张冠李戴=不合格。
3. 如果回答"无法确定"或"抱歉查不到"：仅当检索资料**确实为空或完全不相关**时才视为合格；若检索资料已包含用户问题所涉书籍的篇目（即使不是精确出处），说明检索不充分，应判**不合格**，reason 写明"检索未覆盖准确出处，需反思重检索"。

返回 JSON（不要加任何 markdown 标记）：
{{"passed": true/false, "reason": "一句话说明", "unsupported": ["不实主张1", "不实主张2"]}}"""

_REFLECT_PROMPT = """你是古籍检索诊断专家。一次检索未能找到用户问题的答案，请分析原因。

## 用户问题
{query}

## 检索结果（按相关度排序，可能为空或不相关）
{chunks_summary}

## grounding 校验失败原因
{grounding_reason}

## 任务
诊断为什么检索失败，从以下三类中选：
- **实体抽取错**：关键实体（人名/书名/篇名/术语）未被正确识别
- **Query 过宽**：查询太泛，检索结果噪声大，需要更具体的表述
- **语料缺失**：当前语料库确实不包含该问题的答案

返回 JSON（不要加任何 markdown 标记）：
{{"diagnosis": "实体抽取错/Query过宽/语料缺失", "rewritten_query": "改写后的查询（仅实体抽取错或Query过宽时改写，语料缺失时返回空字符串）"}}"""


def _build_client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)


def _format_chunks_for_grounding(chunks: list[dict]) -> str:
    if not chunks:
        return "（无相关检索资料）"
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {c['text']}\n"
            f"    出处：{c['book']}·{c['chapter']}（{c['version']}）"
        )
    return "\n\n".join(parts)


def _format_chunks_summary(chunks: list[dict]) -> str:
    if not chunks:
        return "（无检索结果）"
    return "\n".join(
        f"  [{i}] {c['book']}·{c['chapter']}（score={c.get('rrf_score', c.get('score', '?'))}）"
        for i, c in enumerate(chunks[:3], 1)
    )


def _extract_json(text: str) -> dict:
    """从 LLM 返回中提取 JSON（兼容 markdown 代码块）。"""
    text = text.strip()
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return json.loads(text)


def check_grounding(answer: str, chunks: list[dict]) -> GroundingResult:
    """LLM-as-judge 校验回答是否被检索 chunk 支撑，同时校验引据合规。

    参数：
        answer: LLM 生成的回答文本
        chunks: 检索到的支撑资料

    返回：
        GroundingResult(passed, reason, unsupported_claims)
    """
    client = _build_client()
    s = get_settings()

    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{
            "role": "user",
            "content": _GROUNDING_PROMPT.format(
                answer=answer,
                chunks_text=_format_chunks_for_grounding(chunks),
            ),
        }],
        temperature=0.0,
        max_tokens=512,
    )

    content = response.choices[0].message.content or "{}"
    try:
        result = _extract_json(content)
    except (json.JSONDecodeError, KeyError):
        return GroundingResult(passed=False, reason=f"JSON 解析失败: {content[:200]}")

    return GroundingResult(
        passed=result.get("passed", False),
        reason=result.get("reason", ""),
        unsupported_claims=result.get("unsupported", []),
    )


def reflect(query: str, chunks: list[dict], grounding_reason: str) -> tuple[str, str]:
    """诊断 why miss + 改写查询。返回 (diagnosis, rewritten_query)。

    参数：
        query:             原始用户查询
        chunks:            当前轮检索结果（可能为空或不相关）
        grounding_reason:  grounding 校验失败原因

    返回：
        (diagnosis, rewritten_query) — diagnosis 为"实体抽取错/Query过宽/语料缺失"之一
    """
    client = _build_client()
    s = get_settings()

    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{
            "role": "user",
            "content": _REFLECT_PROMPT.format(
                query=query,
                chunks_summary=_format_chunks_summary(chunks),
                grounding_reason=grounding_reason,
            ),
        }],
        temperature=0.0,
        max_tokens=512,
    )

    content = response.choices[0].message.content or "{}"
    try:
        result = _extract_json(content)
    except (json.JSONDecodeError, KeyError):
        return ("Query过宽", query)

    return (
        result.get("diagnosis", "Query过宽"),
        result.get("rewritten_query", ""),
    )