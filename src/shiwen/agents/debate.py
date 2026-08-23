"""先贤辩论主图（LangGraph StateGraph）。

S4 Debate Graph：辩题解析 → 循环（仲裁 → 发言 → 漂移监测） → 主持总结。

图结构：
    topic_parse → arbitrate → sage_speak → drift_check → (未满 max_speeches) arbitrate
    drift_check → (满) summarize → END
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from src.shiwen.config import get_settings
from src.shiwen.ingest.embedder import get_embedder

from .drift import DriftMonitor
from .personas import SAGES, SagePersona
from .sage_subgraph import SageSpeech, sage_retrieve, sage_speak
from .urgency import arbitrate, top_speaker


class DebateState(TypedDict, total=False):
    """辩论图共享状态。"""
    topic: str
    user_message: str
    max_speeches: int
    round: int
    speech_log: list[dict]        # [{sage_id, name, school, text, citations, urgency_rank}]
    last_spoken_round: dict[str, int | None]
    urgency_trace: list[dict]     # 每轮紧急度排名
    drift_events: list[dict]      # 漂移事件记录
    drift_monitor: DriftMonitor   # 漂移监测器（带状态，不可序列化但 LangGraph 在内存中运行）
    summary: str
    trace: list[dict]


# ── 节点定义 ────────────────────────────────────────────────────────────────


def _topic_parse(state: DebateState) -> dict:
    """解析辩题，初始化状态。"""
    t0 = time.time()
    trace_entry = {
        "node": "topic_parse",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "topic": state["topic"][:60],
    }
    return {
        "round": 0,
        "speech_log": [],
        "last_spoken_round": {},
        "urgency_trace": [],
        "drift_events": [],
        "drift_monitor": DriftMonitor(),
        "summary": "",
        "trace": [trace_entry],
    }


def _arbitrate_node(state: DebateState) -> dict:
    """紧急度评分仲裁：决定本轮由谁发言。"""
    t0 = time.time()
    sages = list(SAGES.values())
    topic = state["topic"]
    user_message = state.get("user_message", "")

    rnd = state.get("round", 0)
    last_spoken = state.get("last_spoken_round", {})

    # 若用户有追问且辩题较细，可以考虑将 user_message 混入 topic 做相关度
    if user_message and rnd > 0:
        topic_for_arb = f"{topic} {user_message}"
    else:
        topic_for_arb = topic

    results = arbitrate(
        topic=topic_for_arb,
        sages=sages,
        last_spoken_round=last_spoken,
        current_round=rnd,
        user_message=user_message if rnd > 0 else "",
    )

    urgency_trace = state.get("urgency_trace", [])
    urgency_trace.append({
        "round": rnd,
        "ranking": [{"sage_id": r.sage_id, "name": r.name, "total": round(r.total, 4),
                      "relevance": round(r.relevance, 4), "recency": round(r.recency, 4),
                      "rank": r.rank}
                     for r in results],
    })

    trace_entry = {
        "node": "arbitrate",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "top_speaker": results[0].sage_id if results else "",
    }

    return {
        "urgency_trace": urgency_trace,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _sage_speak_node(state: DebateState) -> dict:
    """排名第一的先贤发言。"""
    t0 = time.time()

    urgency_trace = state.get("urgency_trace", [])
    if not urgency_trace:
        return {"trace": state.get("trace", [])}

    latest_ranking = urgency_trace[-1]["ranking"]
    top = latest_ranking[0]
    sage_id = top["sage_id"]
    persona = SAGES[sage_id]

    # 漂移纠偏
    drift_monitor: DriftMonitor = state.get("drift_monitor", DriftMonitor())
    correction_hint = ""
    # 检查本轮前是否有待处理的漂移事件（上一轮触发的）
    drift_events = state.get("drift_events", [])
    if drift_events and drift_events[-1].get("sage_id") == sage_id:
        # 检索该先贤经典语录作为纠偏 anchor
        anchor_chunks = sage_retrieve(persona.stance_hint, persona, top_k=3)
        anchor_texts = [c["text"][:200] for c in anchor_chunks]
        correction_hint = drift_monitor.get_correction_hint(
            sage_id, persona.name, anchor_texts,
        )

    # 发言
    speech: SageSpeech = sage_speak(
        topic=state["topic"],
        persona=persona,
        debate_history=state.get("speech_log", []),
        user_message=state.get("user_message", "") if state.get("round", 0) > 0 else "",
        correction_hint=correction_hint,
    )

    # 更新发言日志
    speech_log = state.get("speech_log", [])
    speech_log.append({
        "sage_id": sage_id,
        "name": persona.name,
        "school": persona.school,
        "text": speech.text,
        "citations": speech.citations,
        "urgency_rank": top["rank"],
    })

    # 更新 last_spoken
    last_spoken = state.get("last_spoken_round", {})
    last_spoken[sage_id] = state.get("round", 0)

    rnd = state.get("round", 0) + 1

    trace_entry = {
        "node": "sage_speak",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "sage_id": sage_id,
        "speech_len": len(speech.text),
        "citations_count": len(speech.citations),
    }

    return {
        "speech_log": speech_log,
        "last_spoken_round": last_spoken,
        "round": rnd,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _drift_check_node(state: DebateState) -> dict:
    """人格漂移软监测。"""
    t0 = time.time()

    drift_monitor: DriftMonitor = state.get("drift_monitor", DriftMonitor())
    speech_log = state.get("speech_log", [])
    if not speech_log:
        return {"trace": state.get("trace", [])}

    latest = speech_log[-1]
    embedder = get_embedder()
    event = drift_monitor.observe(latest["sage_id"], latest["text"], embedder)

    drift_events = state.get("drift_events", [])
    if event:
        drift_events.append(event.to_dict())

    trace_entry = {
        "node": "drift_check",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "drift_detected": event is not None,
    }

    return {
        "drift_events": drift_events,
        "drift_monitor": drift_monitor,
        "trace": state.get("trace", []) + [trace_entry],
    }


def _summarize_node(state: DebateState) -> dict:
    """主持总结：综合各先贤论点，给出辩题总结。"""
    t0 = time.time()

    speech_log = state.get("speech_log", [])
    if not speech_log:
        return {"summary": "辩论未产生发言。", "trace": state.get("trace", [])}

    s = get_settings()
    client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)

    speeches_text = "\n\n".join(
        f"### {h['name']}（{h['school']}）\n{h['text'][:300]}"
        for h in speech_log
    )

    prompt = f"""你是研微（YanWei），一位公正的古籍研究主持人。以下是关于「{state["topic"]}」的先贤辩论记录。

{speeches_text}

请以主持人的身份，撰写一份简洁的辩论总结：
1. 概述各方核心论点（每位先贤一句话概括其立场）
2. 指出主要分歧点
3. 从古籍研究的角度，给出你的学术评价（不偏袒任何一方）

格式：学术总结，客观、严谨。"""

    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )

    summary = response.choices[0].message.content or ""

    trace_entry = {
        "node": "summarize",
        "elapsed_ms": round((time.time() - t0) * 1000),
        "summary_len": len(summary),
    }

    return {
        "summary": summary,
        "trace": state.get("trace", []) + [trace_entry],
    }


# ── 条件路由 ────────────────────────────────────────────────────────────────


def _route_after_speak(state: DebateState) -> str:
    """发言后：发言轮次未满 → 继续仲裁；已满 → 总结。"""
    rnd = state.get("round", 0)
    max_speeches = state.get("max_speeches", 8)
    if rnd < max_speeches:
        return "arbitrate"
    return "summarize"


# ── 构建图 ──────────────────────────────────────────────────────────────────


def build_debate_graph() -> StateGraph:
    """构建编译好的辩论图。

    图结构：
        topic_parse → arbitrate → sage_speak → drift_check
        drift_check → (rnd < max) arbitrate
        drift_check → (rnd >= max) summarize → END
    """
    graph = StateGraph(DebateState)

    graph.add_node("topic_parse", _topic_parse)
    graph.add_node("arbitrate", _arbitrate_node)
    graph.add_node("sage_speak", _sage_speak_node)
    graph.add_node("drift_check", _drift_check_node)
    graph.add_node("summarize", _summarize_node)

    graph.set_entry_point("topic_parse")
    graph.add_edge("topic_parse", "arbitrate")
    graph.add_edge("arbitrate", "sage_speak")
    graph.add_edge("sage_speak", "drift_check")

    graph.add_conditional_edges(
        "drift_check",
        _route_after_speak,
        {"arbitrate": "arbitrate", "summarize": "summarize"},
    )
    graph.add_edge("summarize", END)

    return graph.compile()