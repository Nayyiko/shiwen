"""先贤发言子图：人设 prompt + RAG 检索（book_id 限定该先贤著作）→ 生成发言。

每位先贤发言时，检索范围限定为其本人著作，保证论点可回溯原著。
复用现有检索层（vector + BM25 + RRF），不引入新检索逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from src.shiwen.config import get_settings
from src.shiwen.rag.bm25_store import search as bm25_search
from src.shiwen.rag.fusion import rrf_fuse
from src.shiwen.rag.retriever import RetrievedChunk, retrieve as vector_retrieve

from .personas import SagePersona


@dataclass
class SageSpeech:
    """一位先贤的一次发言。"""
    sage_id: str
    sage_name: str
    text: str
    citations: list[dict]       # 引用的检索 chunk（可回溯）


def sage_retrieve(topic: str, persona: SagePersona,
                  top_k: int = 3) -> list[dict]:
    """RAG 子图检索：限定该先贤著作范围的混合检索。

    topic: 辩题（或当前讨论焦点）
    persona: 先贤人设（books 限定检索范围，book_id 逐个检索后合并）
    """
    all_hits: list[dict] = []
    seen_ids: set[str] = set()

    for book_id in persona.books:
        # 向量检索
        vector_hits = vector_retrieve(topic, top_k=top_k, book_id=book_id)
        for c in vector_hits:
            if c.id not in seen_ids:
                seen_ids.add(c.id)
                all_hits.append(_chunk_to_dict(c))

        # BM25 检索
        bm25_hits = bm25_search(topic, top_k=top_k, book_id=book_id)
        for h in bm25_hits:
            if h["id"] not in seen_ids:
                seen_ids.add(h["id"])
                all_hits.append(h)

    # RRF 融合
    if not all_hits:
        return []

    vector_dicts = [h for h in all_hits if "score" in h and isinstance(h["score"], float)]
    bm25_dicts = [h for h in all_hits if "bm25_score" in h]
    if vector_dicts and bm25_dicts:
        return rrf_fuse(vector_dicts, bm25_dicts, top_k=top_k)
    elif vector_dicts:
        return sorted(vector_dicts, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
    else:
        return sorted(bm25_dicts, key=lambda x: x.get("bm25_score", 0), reverse=True)[:top_k]


def sage_speak(
    topic: str,
    persona: SagePersona,
    debate_history: list[dict] | None = None,
    user_message: str = "",
    correction_hint: str = "",
    top_k: int = 3,
) -> SageSpeech:
    """先贤发言：检索本人著作 → 人设 prompt 生成发言。

    Args:
        topic: 辩题
        persona: 先贤人设
        debate_history: 之前的发言记录 [{sage, text}, ...]
        user_message: 用户最近的追问/插话
        correction_hint: 漂移纠偏提示（空字符串=无漂移）
        top_k: 检索 chunk 数量

    Returns:
        SageSpeech（发言文本 + 引用溯源）
    """
    # 检索：只查该先贤本人的著作
    chunks = sage_retrieve(topic, persona, top_k=top_k)

    # 构建生成 prompt
    s = get_settings()
    client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)

    # 历史上下文
    history_text = ""
    if debate_history:
        recent = debate_history[-4:]  # 最近 4 轮
        history_text = "\n".join(
            f"【{h['name']}（{h['school']}）】{h['text'][:200]}"
            for h in recent
        )

    # 检索资料
    chunks_text = "\n\n".join(
        f"[{i}] {c['text'][:300]}\n    出处：{c['book']}·{c['chapter']}（{c['version']}）"
        for i, c in enumerate(chunks, 1)
    ) if chunks else "（未检索到相关原文，请基于你的学派立场即兴发言，但不要编造具体引文）"

    # 用户消息
    user_text = f"\n\n## 用户追问\n{user_message}" if user_message else ""

    prompt = f"""## 辩题
{topic}

## 辩论进程
{history_text if history_text else "（辩论刚开始，你是首位发言者）"}
{user_text}

## 你的著作中与此辩题相关的原文
{chunks_text}

请基于以上原文，以{persona.name}的身份发表你的论点。严格遵循你的人设，引用原文时必须标注出处。"""

    response = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[
            {"role": "system", "content": persona.persona_prompt + correction_hint},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,  # 辩论需要一定创造力，但不宜过高（会编造）
        max_tokens=512,
    )

    text = response.choices[0].message.content or ""

    return SageSpeech(
        sage_id=persona.id,
        sage_name=persona.name,
        text=text,
        citations=chunks,
    )


def _chunk_to_dict(c: RetrievedChunk) -> dict:
    return {
        "id": c.id, "text": c.text, "score": c.score,
        "book_id": c.book_id, "book": c.book, "author": c.author,
        "dynasty": c.dynasty, "category": c.category, "version": c.version,
        "part": c.part, "chapter": c.chapter,
        "chapter_index": c.chapter_index, "chunk_index": c.chunk_index,
    }