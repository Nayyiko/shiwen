"""S5 写作评测脚本：3 个选题 → 引据可回溯率 + 书目覆盖率。

运行：
  docker compose run --rm backend bash -c "pip install httpx -q && python eval/run_writing.py"
  docker compose run --rm backend bash -c "pip install httpx -q && python eval/run_writing.py --limit 1"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shiwen.config import get_settings
from src.shiwen.writing.graph import build_writing_graph
from src.shiwen.writing.citations import verify_citations

CHECKPOINT = Path("eval/writing_results.json")


def load_topics(path: str = "eval/writing.yaml") -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["writing_topics"]


def evaluate(limit: int | None = None) -> dict:
    topics = load_topics()
    if limit:
        topics = topics[:limit]

    graph = build_writing_graph()
    results: list[dict] = []
    total_citation_rate = 0.0
    total_book_coverage = 0.0
    total_elapsed = 0.0

    for i, t in enumerate(topics):
        tid = t["id"]
        topic = t["topic"]
        expected_books = set(t["expected_books"])

        print(f"[{i+1}/{len(topics)}] {tid} 「{topic}」...", flush=True)

        t0 = time.time()
        result = graph.invoke({
            "topic": topic,
            "max_sections": t.get("min_sections", 3),
            "outline": [],
            "section_index": 0,
            "all_chunks": [],
            "article": "",
            "trace": [],
        })
        elapsed = time.time() - t0
        total_elapsed += elapsed

        article = result.get("article", "")
        all_chunks = result.get("all_chunks", [])
        outline = result.get("outline", [])

        # 引据可回溯率
        verify = verify_citations(article, all_chunks)
        citation_rate = verify["rate"]
        total_citation_rate += citation_rate

        # 书目覆盖率：引用的书中有多少是 expected_books
        cited_books = set()
        for c in verify.get("matched_list", []):
            # need book_id from chunk_pool matching citation
            book = c["book"]
            for ch in all_chunks:
                if ch.get("book") == book:
                    cited_books.add(ch.get("book_id", book))
                    break

        book_coverage = len(cited_books & expected_books) / len(cited_books) if cited_books else 1.0
        total_book_coverage += book_coverage

        sections_info = []
        for sec in outline:
            sections_info.append({
                "title": sec.get("title", ""),
                "text_len": len(sec.get("text", "")),
                "chunks": len(sec.get("chunks", [])),
            })

        results.append({
            "id": tid,
            "topic": topic,
            "sections": sections_info,
            "citation_total": verify["total"],
            "citation_matched": verify["matched"],
            "citation_rate": round(citation_rate, 4),
            "unmatched": verify["unmatched"],
            "book_coverage": round(book_coverage, 4),
            "cited_books": sorted(cited_books),
            "expected_books": sorted(expected_books),
            "article_len": len(article),
            "article_preview": article[:300],
            "elapsed_s": round(elapsed, 1),
        })

        print(f"  citations={verify['matched']}/{verify['total']} "
              f"({citation_rate:.0%}) "
              f"books={book_coverage:.0%} "
              f"{elapsed:.1f}s")

        CHECKPOINT.write_text(json.dumps(
            {"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(results)
    metrics = {
        "total_topics": n,
        "avg_citation_traceability": round(total_citation_rate / n, 4) if n else 0,
        "avg_book_coverage": round(total_book_coverage / n, 4) if n else 0,
        "avg_elapsed_s": round(total_elapsed / n, 1) if n else 0,
        "total_elapsed_s": round(total_elapsed, 1),
    }

    return {"metrics": metrics, "results": results}


def write_report(report: dict, path: str = "eval/report_writing.md") -> None:
    m = report["metrics"]
    lines = [
        "# 识文新裁 S5 写作评测报告",
        "",
        "## 概览",
        f"- 选题数：{m['total_topics']}",
        f"- 总耗时：{m['total_elapsed_s']}s（平均 {m['avg_elapsed_s']}s/题）",
        "",
        "## 指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| **引据可回溯率** | {m['avg_citation_traceability']:.0%} | 文章引据标签 ∈ 检索池 citation 集合（确定性） |",
        f"| **书目覆盖率** | {m['avg_book_coverage']:.0%} | 引用书目属于 expected_books 的比例 |",
        "",
        "## 各题详情",
        "",
        "| ID | 选题 | 节数 | 引据(命中/总数) | 可回溯率 | 书目覆盖率 | 延迟 |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in report["results"]:
        lines.append(
            f"| {r['id']} | {r['topic'][:25]} | {len(r['sections'])} | "
            f"{r['citation_matched']}/{r['citation_total']} | "
            f"{r['citation_rate']:.0%} | "
            f"{r['book_coverage']:.0%} | "
            f"{r['elapsed_s']}s |"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S5 写作评测")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="eval/report_writing.md")
    args = parser.parse_args()

    if not get_settings().deepseek_api_key:
        print("[eval] ❌ DEEPSEEK_API_KEY 未配置")
        return

    print(f"[eval] 加载写作评测集...")
    topics = load_topics()
    print(f"[eval] 共 {len(topics)} 个选题")

    print(f"[eval] 开始评测...\n")
    report = evaluate(limit=args.limit)
    write_report(report, path=args.output)

    m = report["metrics"]
    print(f"\n[eval] 完成：")
    print(f"  引据可回溯率 = {m['avg_citation_traceability']:.0%}")
    print(f"  书目覆盖率   = {m['avg_book_coverage']:.0%}")
    print(f"  平均延迟     = {m['avg_elapsed_s']}s")


if __name__ == "__main__":
    main()