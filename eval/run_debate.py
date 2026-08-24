"""S4 辩论评测脚本：5 个辩题 → 路由正确率 + 论据引用率 + 漂移记录。

运行：
  docker compose run --rm backend bash -c "pip install httpx -q && python eval/run_debate.py"
  docker compose run --rm backend bash -c "pip install httpx -q && python eval/run_debate.py --limit 2"
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
from src.shiwen.agents.debate import build_debate_graph
from src.shiwen.agents.personas import SAGES

CHECKPOINT = Path("eval/debate_results.json")


def load_topics(path: str = "eval/debate.yaml") -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["debate_topics"]


def evaluate(limit: int | None = None) -> dict:
    """运行辩论评测，返回汇总指标。"""
    topics = load_topics()
    if limit:
        topics = topics[:limit]

    graph = build_debate_graph()

    results: list[dict] = []
    total_routing_ok = 0
    total_citation_ok = 0
    total_citation_count = 0
    total_drift_events = 0
    total_elapsed = 0.0

    for i, t in enumerate(topics):
        tid = t["id"]
        topic = t["topic"]
        expected = set(t["expected_sages"])

        print(f"[{i+1}/{len(topics)}] {tid} 「{topic}」...", flush=True)

        t0 = time.time()
        result = graph.invoke({
            "topic": topic,
            "user_message": "",
            "max_speeches": 4,
            "round": 0,
            "speech_log": [],
            "last_spoken_round": {},
            "urgency_trace": [],
            "drift_events": [],
            "summary": "",
            "trace": [],
        })
        elapsed = time.time() - t0
        total_elapsed += elapsed

        speech_log = result.get("speech_log", [])
        urgency_trace = result.get("urgency_trace", [])

        # 路由正确率：expected_sages 是否在前 2 位发言
        first_two = set()
        if urgency_trace:
            ranking = urgency_trace[0].get("ranking", [])
            first_two = {r["sage_id"] for r in ranking[:2]}
        routing_ok = bool(first_two & expected)
        if routing_ok:
            total_routing_ok += 1

        # 论据引用率：每次发言引用的 chunk book 是否属于该先贤著作
        speeches_citation = []
        for h in speech_log:
            sage_id = h["sage_id"]
            persona = SAGES.get(sage_id)
            sage_books = set(persona.books) if persona else set()
            citations = h.get("citations", [])
            valid = 0
            for c in citations:
                if c.get("book_id") in sage_books:
                    valid += 1
            rate = valid / len(citations) if citations else 1.0
            total_citation_ok += valid
            total_citation_count += len(citations)
            speeches_citation.append({
                "sage_id": sage_id,
                "name": h["name"],
                "citations_total": len(citations),
                "citations_valid": valid,
                "rate": round(rate, 4),
            })

        # 漂移事件
        drift_events = result.get("drift_events", [])
        total_drift_events += len(drift_events)

        results.append({
            "id": tid,
            "topic": topic,
            "routing_ok": routing_ok,
            "expected": sorted(expected),
            "first_two": sorted(first_two),
            "speeches_count": len(speech_log),
            "speeches_citation": speeches_citation,
            "drift_count": len(drift_events),
            "summary_preview": result.get("summary", "")[:200],
            "elapsed_s": round(elapsed, 1),
        })

        print(f"  routing={'✓' if routing_ok else '✗'} "
              f"speeches={len(speech_log)} "
              f"drift={len(drift_events)} "
              f"{elapsed:.1f}s")

        CHECKPOINT.write_text(json.dumps(
            {"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(results)
    metrics = {
        "total_topics": n,
        "routing_accuracy": round(total_routing_ok / n, 4) if n else 0,
        "citation_compliance": round(total_citation_ok / total_citation_count, 4)
        if total_citation_count else 0,
        "total_citations": total_citation_count,
        "valid_citations": total_citation_ok,
        "total_drift_events": total_drift_events,
        "avg_elapsed_s": round(total_elapsed / n, 1) if n else 0,
        "total_elapsed_s": round(total_elapsed, 1),
    }

    return {"metrics": metrics, "results": results}


def write_report(report: dict, path: str = "eval/report_debate.md") -> None:
    m = report["metrics"]
    lines = [
        "# 识文新裁 S4 辩论评测报告",
        "",
        "## 概览",
        f"- 辩论题数：{m['total_topics']}",
        f"- 总耗时：{m['total_elapsed_s']}s（平均 {m['avg_elapsed_s']}s/题）",
        "",
        "## 指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| **辩题路由正确率** | {m['routing_accuracy']:.0%} | expected_sages 出现在前两位发言 |",
        f"| **论据引用率** | {m['valid_citations']}/{m['total_citations']} = {m['citation_compliance']:.0%} | 发言引用 chunk 的 book 属于该先贤著作 |",
        f"| **漂移事件数** | {m['total_drift_events']} | 检测到的人格漂移次数 |",
        "",
        "## 各题详情",
        "",
        "| ID | 辩题 | 路由 | 发言数 | 引用率 | 漂移 | 延迟 |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in report["results"]:
        rates = [s["rate"] for s in r["speeches_citation"]]
        avg_rate = sum(rates) / len(rates) if rates else 0
        lines.append(
            f"| {r['id']} | {r['topic'][:30]} | "
            f"{'✓' if r['routing_ok'] else '✗'} | "
            f"{r['speeches_count']} | "
            f"{avg_rate:.0%} | "
            f"{r['drift_count']} | "
            f"{r['elapsed_s']}s |"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S4 辩论评测")
    parser.add_argument("--limit", type=int, default=None, help="限制辩论题数")
    parser.add_argument("--output", type=str, default="eval/report_debate.md")
    args = parser.parse_args()

    if not get_settings().deepseek_api_key:
        print("[eval] ❌ DEEPSEEK_API_KEY 未配置，跳过辩论评测（无法生成发言）")
        return

    print(f"[eval] 加载辩论评测集...")
    topics = load_topics()
    print(f"[eval] 共 {len(topics)} 个辩题")

    print(f"[eval] 开始评测...\n")
    report = evaluate(limit=args.limit)
    write_report(report, path=args.output)

    m = report["metrics"]
    print(f"\n[eval] 完成：")
    print(f"  辩题路由正确率 = {m['routing_accuracy']:.0%}")
    print(f"  论据引用率     = {m['valid_citations']}/{m['total_citations']} = {m['citation_compliance']:.0%}")
    print(f"  漂移事件数     = {m['total_drift_events']}")
    print(f"  平均延迟       = {m['avg_elapsed_s']}s")


if __name__ == "__main__":
    main()