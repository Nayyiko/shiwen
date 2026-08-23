"""生成层：DeepSeek API 调用，带严格引据约束的 system prompt。

S2 单跳 RAG 的生成节点：检索结果 + 用户问题 → LLM → 带引据的回答。
"""

from __future__ import annotations

from openai import OpenAI

from src.shiwen.config import get_settings
from .retriever import RetrievedChunk

_SYSTEM_PROMPT = """你是研微（YanWei），一位严谨的古籍研究助手。你的回答必须基于提供的检索资料，并严格遵守以下规则：

1. **只引用提供的资料**：每一条事实主张都必须能在提供的检索资料中找到原文支撑。如果资料不足以回答，明确说"根据现有资料，暂无法确定"——绝不可编造。
2. **标注引据**：每引用一条原文，必须标注出处，格式为「书名·篇名（版本）」，例如「论语·学而篇（通行本）」。
3. **回答简洁准确**：先直接回答问题，再提供原文佐证与简要解释。
4. **无资料时诚实**：检索资料为空或不相关时，回复"抱歉，当前语料库中未找到相关资料，无法回答此问题。"不要猜测。

你是学术助手，不是文艺创作者——保持客观、严谨、可验证。"""

_USER_TEMPLATE = """## 用户问题
{query}

## 检索资料（按相关度排序）
{chunks_text}

请基于以上资料回答用户问题。"""


def _build_client() -> OpenAI:
    s = get_settings()
    if not s.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY 未配置，请在 .env 中填入 DeepSeek API Key")
    return OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    """将检索结果格式化为 LLM prompt 中的资料卡片。"""
    if not chunks:
        return "（无相关检索资料）"

    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {c.text}\n"
            f"    出处：{c.citation}  作者：{c.author}（{c.dynasty}）"
        )
    return "\n\n".join(parts)


def generate(query: str, chunks: list[RetrievedChunk]) -> str:
    """调用 DeepSeek 生成带引据的回答。

    参数：
        query:  用户原始问题
        chunks: 检索到的相关资料（含元数据）

    返回：
        LLM 生成的回答文本（含引据标注）

    异常：
        ValueError: 未配置 API key
    """
    client = _build_client()
    s = get_settings()

    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(
                query=query,
                chunks_text=_format_chunks(chunks),
            )},
        ],
        temperature=0.3,  # 事实性任务，低温度
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""