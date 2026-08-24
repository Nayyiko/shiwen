"""S6 新裁角色扮演图（LangGraph StateGraph）。

与 S4 辩论图不同：S6 是 1v1 对话，用户指定一位先贤，多轮沉浸式交互。
图结构简单：retrieve → generate → END，多轮由调用方传 history 驱动。
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from src.shiwen.agents.personas import SAGES, SagePersona
from src.shiwen.agents.sage_subgraph import sage_retrieve
from src.shiwen.config import get_settings


class RoleplayState(TypedDict, total=False):
    """角色扮演图共享状态。"""
    sage_id: str                  # 先贤 ID（kongzi/mengzi/laozi/hanfei）
    user_message: str             # 用户当前消息
    history: list[dict]           # 对话历史 [{role, content}, ...]
    chunks: list[dict]            # 检索到的原文
    response: str                 # 先贤回复
    trace: list[dict]             # 节点耗时


# ── 节点定义 ──────────────────────────────────────────────────────────────────


def _retrieve_node(state: RoleplayState) -> dict:
    """检索节点：在指定先贤著作中检索与用户消息最相关的原文。"""
    t0 = time.time()
    sage_id = state["sage_id"]
    persona = SAGES.get(sage_id)
    if persona is None:
        return {
            "chunks": [],
            "trace": [{"node": "retrieve", "elapsed_ms": round((time.time() - t0) * 1000)}],
        }

    chunks = sage_retrieve(state["user_message"], persona, top_k=3)
    return {
        "chunks": chunks,
        "trace": [{"node": "retrieve", "elapsed_ms": round((time.time() - t0) * 1000)}],
    }


def _generate_node(state: RoleplayState) -> dict:
    """生成节点：基于人设 + 检索原文 + 对话历史，生成先贤回复。"""
    t0 = time.time()
    sage_id = state["sage_id"]
    persona = SAGES.get(sage_id)
    if persona is None:
        return {
            "response": "（这位先贤尚未就座，请另选一位。）",
            "trace": [*state.get("trace", []),
                       {"node": "generate", "elapsed_ms": round((time.time() - t0) * 1000)}],
        }

    s = get_settings()
    client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)

    # 历史上下文
    history_text = ""
    history = state.get("history", [])
    if history:
        recent = history[-6:]  # 最近 6 轮
        history_text = "\n".join(
            f"【{'用户' if h['role'] == 'user' else persona.name}】{h['content'][:300]}"
            for h in recent
        )

    # 检索资料
    chunks = state.get("chunks", [])
    chunks_text = "\n\n".join(
        f"[{i}] {c['text'][:300]}\n    出处：{c['book']}·{c['chapter']}（{c['version']}）"
        for i, c in enumerate(chunks, 1)
    ) if chunks else "（未检索到相关原文，请基于你的学派立场回应，但不要编造具体引文）"

    user_message = state["user_message"]

    prompt = f"""## 你的身份
你是{persona.name}（{persona.dynasty}时期{persona.school}代表人物）。

## 对话历史
{history_text if history_text else "（这是对话的开始，用户刚刚向你打招呼）"}

## 用户刚刚说的话
{user_message}

## 你的著作中与话题相关的原文
{chunks_text}

请以{persona.name}的身份回复用户。要求：
1. 严格遵循你的人设——学派立场、语言风格、用典习惯都必须与{persona.name}一致。
2. 如果检索资料中有相关原文，必须引用并标注出处，格式：「书名·篇名（版本）」。
3. 回复应自然如对话，不要像辩论发言——你在与一位求教者交谈，而非与对手辩论。
4. 回复长度控制在 100-200 字，言简意赅。
5. 不要出现"作为{persona.name}""根据检索资料"等元描述。"""

    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[
            {"role": "system", "content": persona.persona_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=400,
    )

    text = response.choices[0].message.content or ""

    return {
        "response": text,
        "trace": [*state.get("trace", []),
                   {"node": "generate", "elapsed_ms": round((time.time() - t0) * 1000)}],
    }


# ── 图构建 ────────────────────────────────────────────────────────────────────


def build_roleplay_graph() -> StateGraph:
    """构建角色扮演 LangGraph。

    retrieve → generate → END
    """
    graph = StateGraph(RoleplayState)

    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()