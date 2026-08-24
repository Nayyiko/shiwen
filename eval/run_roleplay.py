"""S6 角色扮演评测脚本：12 个问题（4 位先贤 × 3）→ 引据合规率 + 人设关键词命中。

运行：
  docker compose run --rm backend bash -c "pip install httpx -q && python eval/run_roleplay.py"
  docker compose run --rm backend bash -c "pip install httpx -q && python eval/run_roleplay.py --limit 3"
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
from src.shiwen.roleplay.graph import build_roleplay_graph
from src.shiwen.writing.citations import verify_citations

CHECKPOINT = Path("eval/roleplay_results.json")


def load_cases(path: str = "eval/roleplay.yaml") -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["roleplay_cases"]


def evaluate(limit: int | None = None) -> dict:
    cases = load_cases()
    if limit:
        cases = cases[:limit]

    graph = build_roleplay_graph()
    results: list[dict] = []
    total_citation_rate = 0.0
    total_persona_rate = 0.0
    total_elapsed = 0.0

    for i, c in enumerate(cases):
        cid = c["id"]
        sage_id = c["sage_id"]
        message = c["message"]
        expected_keywords = c["persona_keywords"]

        print(f"[{i+1}/{len(cases)}] {cid} {sage_id} 「{message}」...", flush=True)

        t0 = time.time()
        result = graph.invoke({
            "sage_id": sage_id,
            "user_message": message,
            "history": [],
            "chunks": [],
            "response": "",
            "trace": [],
        })
        elapsed = time.time() - t0
        total_elapsed += elapsed

        response = result.get("response", "")
        chunks = result.get("chunks", [])

        # 引据合规率
        verify = verify_citations(response, chunks)
        citation_total = verify["total"]
        citation_matched = verify["matched"]
        citation_rate = verify["rate"]
        total_citation_rate += citation_rate

        # 人设关键词命中率
        persona_hits = sum(1 for kw in expected_keywords if kw in response)
        persona_rate = persona_hits / len(expected_keywords) if expected_keywords else 1.0
        total_persona_rate += persona_rate

        print(f"  citations={citation_matched}/{citation_total} "
              f"({citation_rate:.0%}) "
              f"persona={persona_hits}/{len(expected_keywords)} "
              f"({persona_rate:.0%}) "
              f"{elapsed:.1f}s")

        results.append({
            "id": cid,
            "sage_id": sage_id,
            "message": message,
            "response_len": len(response),
            "response_preview": response[:200],
            "citation_total": citation_total,
            "citation_matched": citation_matched,
            "citation_rate": round(citation_rate, 4),
            "unmatched": verify["unmatched"],
            "persona_keywords_hit": persona_hits,
            "persona_keywords_total": len(expected_keywords),
            "persona_rate": round(persona_rate, 4),
            "elapsed_s": round(elapsed, 1),
        })

        CHECKPOINT.write_text(json.dumps(
            {"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(results)
    metrics = {
        "total_cases": n,
        "avg_citation_traceability": round(total_citation_rate / n, 4) if n else 0,
        "avg_persona_keyword_rate": round(total_persona_rate / n, 4) if n else 0,
        "avg_elapsed_s": round(total_elapsed / n, 1) if n else 0,
        "total_elapsed_s": round(total_elapsed, 1),
    }

    return {"metrics": metrics, "results": results}


def write_report(report: dict, path: str = "eval/report_roleplay.md") -> None:
    m = report["metrics"]
    lines = [
        "# 识文新裁 S6 角色扮演评测报告",
        "",
        "## 概览",
        f"- 评测用例数：{m['total_cases']}",
        f"- 总耗时：{m['total_elapsed_s']}s（平均 {m['avg_elapsed_s']}s/题）",
        "",
        "## 指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| **引据合规率** | {m['avg_citation_traceability']:.0%} | 回复中引据可回溯到检索池（确定性） |",
        f"| **人设关键词命中率** | {m['avg_persona_keyword_rate']:.0%} | 回复包含学派核心关键词的比例 |",
        "",
        "## 各题详情",
        "",
        "| ID | 先贤 | 问题 | 引据(命中/总数) | 合规率 | 关键词(命中/总数) | 命中率 | 延迟 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in report["results"]:
        lines.append(
            f"| {r['id']} | {r['sage_id']} | {r['message'][:20]} | "
            f"{r['citation_matched']}/{r['citation_total']} | "
            f"{r['citation_rate']:.0%} | "
            f"{r['persona_keywords_hit']}/{r['persona_keywords_total']} | "
            f"{r['persona_rate']:.0%} | "
            f"{r['elapsed_s']}s |"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S6 角色扮演评测")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="eval/report_roleplay.md")
    args = parser.parse_args()

    if not get_settings().deepseek_api_key:
        print("[eval] ❌ DEEPSEEK_API_KEY 未配置")
        return

    print(f"[eval] 加载角色扮演评测集...")
    cases = load_cases()
    print(f"[eval] 共 {len(cases)} 个用例")

    print(f"[eval] 开始评测...\n")
    report = evaluate(limit=args.limit)
    write_report(report, path=args.output)

    m = report["metrics"]
    print(f"\n[eval] 完成：")
    print(f"  引据合规率     = {m['avg_citation_traceability']:.0%}")
    print(f"  人设关键词命中率 = {m['avg_persona_keyword_rate']:.0%}")
    print(f"  平均延迟       = {m['avg_elapsed_s']}s")


if __name__ == "__main__":
    main()