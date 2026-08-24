"""S8 MCP 工具层：把检索 / 引据校验 / 人物关系 / 写作引擎封装为 MCP 工具。

对外暴露 4 个工具：
  - retrieve: 三路混合检索（向量 + BM25 + RRF）
  - verify: 引据合规校验
  - query_person: 人物关系查询
  - write: 研究写作引擎

运行（stdio 模式，供 Claude Code 或其他 MCP 客户端连接）：
  python -m src.shiwen.mcp.server

也可以直接 import 使用：
  from src.shiwen.mcp.server import mcp
"""

from __future__ import annotations

import json

from fastmcp import FastMCP

mcp = FastMCP("识文新裁")


# ── Tool 1: 检索 ──────────────────────────────────────────────────────────────


@mcp.tool()
def retrieve(query: str, top_k: int = 5, book_id: str | None = None) -> str:
    """古籍混合检索：向量 + BM25 + RRF 融合。

    Args:
        query: 检索问题（如"仁是什么"）
        top_k: 返回 chunk 数量
        book_id: 可选，按书目过滤（如 lunyu）

    Returns:
        JSON 字符串，包含检索到的 chunk 列表（每条含 text/book/chapter/version 等）
    """
    from src.shiwen.rag.retriever import retrieve as _retrieve

    chunks = _retrieve(query, top_k=top_k, book_id=book_id)
    results = [
        {
            "id": c.id,
            "text": c.text,
            "book": c.book,
            "book_id": c.book_id,
            "author": c.author,
            "dynasty": c.dynasty,
            "category": c.category,
            "chapter": c.chapter,
            "version": c.version,
            "score": c.score,
        }
        for c in chunks
    ]
    return json.dumps({"results": results}, ensure_ascii=False)


# ── Tool 2: 引据校验 ──────────────────────────────────────────────────────────


@mcp.tool()
def verify(text: str, chunks_json: str) -> str:
    """校验 text 中的引据是否可回溯到 chunk 检索池。

    Args:
        text: 待校验文本（可含多个「书名·篇名（版本）」格式引据）
        chunks_json: JSON 字符串，chunk 列表（与 retrieve 返回格式兼容）

    Returns:
        JSON 字符串：{"total", "matched", "rate", "unmatched", "matched_list"}
    """
    from src.shiwen.writing.citations import verify_citations

    chunks = json.loads(chunks_json).get("results", [])
    result = verify_citations(text, chunks)
    # convert unmatched/matched_list for JSON
    return json.dumps({
        "total": result["total"],
        "matched": result["matched"],
        "rate": result["rate"],
        "unmatched": result["unmatched"],
        "matched_list": result["matched_list"],
    }, ensure_ascii=False)


# ── Tool 3: 人物关系查询 ──────────────────────────────────────────────────────


@mcp.tool()
def query_person(name_or_id: str) -> str:
    """查询人物及其著作、关系。

    Args:
        name_or_id: 人物姓名（如"孔子"）或人物 ID（如"kongzi"）

    Returns:
        JSON 字符串：{"person": {...}, "works": [...], "relations": [...]}
    """
    from src.shiwen.ingest.people import query_person as _query_person

    result = _query_person(name_or_id)
    if not result:
        return json.dumps({"error": f"未找到人物：{name_or_id}"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


# ── Tool 4: 研究写作 ──────────────────────────────────────────────────────────


@mcp.tool()
def write(topic: str, max_sections: int = 4) -> str:
    """研究写作引擎：选题 → 大纲 → 逐节检索 → 逐节写作 → 综合润色。

    Args:
        topic: 写作选题（如"论语中的仁学思想"）
        max_sections: 最大节数

    Returns:
        JSON 字符串：{"article", "sections", "citations", "trace"}
    """
    from src.shiwen.writing.graph import build_writing_graph

    graph = build_writing_graph()
    result = graph.invoke({
        "topic": topic,
        "max_sections": max_sections,
        "outline": [],
        "section_index": 0,
        "all_chunks": [],
        "article": "",
        "trace": [],
    })

    sections = [
        {"title": sec.get("title", ""), "text": sec.get("text", "")}
        for sec in result.get("outline", [])
    ]
    citations = [
        {
            "book": c.get("book", ""),
            "chapter": c.get("chapter", ""),
            "version": c.get("version", ""),
            "text": c.get("text", "")[:200],
        }
        for c in result.get("all_chunks", [])[:10]
    ]

    return json.dumps({
        "article": result.get("article", ""),
        "sections": sections,
        "citations": citations,
        "trace": result.get("trace", []),
    }, ensure_ascii=False)


# ── 入口 ──────────────────────────────────────────────────────────────────────


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
