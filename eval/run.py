"""S3 评测脚本：100 题评测集 → Recall@5 / MRR / 引据合规率 / Faithfulness。

运行：
  docker compose run --rm backend python eval/run.py
  docker compose run --rm backend python eval/run.py --limit 10  # 仅跑前 10 题
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# 允许 `python eval/run.py` 直接运行：脚本所在目录是 eval/，项目根不在 sys.path 上，
# 手动把项目根加入，否则 `from src.shiwen...` 会 ModuleNotFoundError。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shiwen.config import get_settings
from src.shiwen.rag.graph import build_graph

CHECKPOINT = Path("eval/results.json")


def load_questions(path: str = "eval/questions.yaml") -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["questions"]


def load_checkpoint() -> dict[str, dict]:
    """读断点：qid → 已完成结果。"""
    if not CHECKPOINT.exists():
        return {}
    try:
        data = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        return {r["id"]: r for r in data["results"]}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _faithfulness_judge(answer: str, chunks: list[dict], query: str) -> tuple[int, str]:
    """LLM-as-judge：忠实度评分 0-5。

    0 = 完全编造，与检索资料无关
    5 = 所有主张均有检索资料原文支撑，引据标注准确
    """
    from openai import OpenAI

    s = get_settings()
    client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)

    chunks_text = "\n\n".join(
        f"[{i}] {c['text'][:300]}\n    出处：{c['book']}·{c['chapter']}（{c['version']}）"
        for i, c in enumerate(chunks, 1)
    ) if chunks else "（无检索资料）"

    prompt = f"""你是古籍引据忠实度评审员。请评估以下回答是否忠实于检索资料，并给出 0-5 分。

## 用户问题
{query}

## 检索资料
{chunks_text}

## 回答
{answer}

## 评分标准
- 5：所有事实主张均有检索资料原文支撑，引据标注准确，无编造
- 4：大部分有支撑，个别细节不在检索结果中但不影响主体
- 3：主体正确但有部分编造或引据错误
- 2：仅有少量正确信息，大部分与检索资料无关
- 1：回答几乎不依赖检索资料，或大量编造
- 0：完全编造，或回答与问题无关

返回 JSON（不要加 markdown 标记）：
{{"score": 0-5, "reason": "一句话说明"}}"""

    resp = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        import re
        m = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        result = json.loads(m.group(0)) if m else json.loads(content)
        return result.get("score", 0), result.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        return 0, f"JSON 解析失败: {content[:200]}"


def evaluate(limit: int | None = None, resume: bool = False) -> dict:
    """运行评测，返回汇总指标。

    resume=True 时跳过断点中已完成的问题，从上次中断处继续。
    """
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    done: dict[str, dict] = load_checkpoint() if resume else {}
    if resume and done:
        print(f"[eval] 断点续跑：已跳过 {len(done)} 题")

    graph = build_graph()

    results: list[dict] = list(done.values())
    total_recall = 0           # golden book_id 出现在 Top-5 中的次数
    total_mrr = 0.0            # 首个 golden 命中的倒数排名之和
    total_citation_ok = 0      # 回答引据与 golden 一致的次数
    total_faithfulness = 0.0   # 忠实度评分之和
    total_elapsed = 0.0
    passed = 0                 # grounding 通过次数

    for i, q in enumerate(questions):
        qid = q["id"]
        if qid in done:
            continue
        qtype = q["type"]
        query = q["query"]
        golden_book = q["golden_book_id"]
        golden_chapter = q.get("golden_chapter", "")

        print(f"[{i+1}/{len(questions)}] {qid} [{qtype}] {query[:60]}...", end=" ", flush=True)

        t0 = time.time()
        result = graph.invoke({
            "query": query,
            "book_id": None,
            "category": None,
            "round": 0,
            "max_rounds": 3,
            "chunks": [],
            "answer": "",
            "grounding_pass": False,
            "grounding_reason": "",
            "diagnosis": "",
            "rewritten_query": "",
            "trace": [],
        })
        elapsed = time.time() - t0
        total_elapsed += elapsed

        chunks = result.get("chunks", [])
        answer = result.get("answer", "")
        grounding_pass = result.get("grounding_pass", False)
        rounds = result.get("round", 0) + 1

        # Recall@5: golden book_id 出现在 Top-5 中？
        book_hits = [c for c in chunks if c["book_id"] == golden_book]
        recall_hit = len(book_hits) > 0
        if recall_hit:
            total_recall += 1

        # MRR: 首个 golden book_id 的倒数排名
        first_rank = None
        for rank, c in enumerate(chunks, 1):
            if c["book_id"] == golden_book:
                first_rank = rank
                break
        if first_rank:
            total_mrr += 1.0 / first_rank

        # 引据合规：golden_chapter 前缀匹配（仅对考据/翻译类）
        citation_ok = False
        if golden_chapter:
            chapter_hits = [
                c for c in chunks
                if c["book_id"] == golden_book
                and (c["chapter"] or "").startswith(golden_chapter)
            ]
            citation_ok = len(chapter_hits) > 0
            if citation_ok:
                total_citation_ok += 1

        # Faithfulness: LLM-as-judge
        faith_score, faith_reason = _faithfulness_judge(answer, chunks, query)
        total_faithfulness += faith_score

        if grounding_pass:
            passed += 1

        results.append({
            "id": qid, "type": qtype, "query": query,
            "recall": recall_hit,
            "mrr_rank": first_rank,
            "citation_ok": citation_ok,
            "faithfulness": faith_score,
            "faithfulness_reason": faith_reason,
            "grounding_pass": grounding_pass,
            "rounds": rounds,
            "elapsed_s": round(elapsed, 1),
            "answer_preview": answer[:200],
        })

        # 逐题断点：崩溃后可 --resume 续跑
        CHECKPOINT.write_text(json.dumps(
            {"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"recall={'✓' if recall_hit else '✗'} "
              f"MRR={1.0/first_rank if first_rank else 0:.3f} "
              f"cite={'✓' if citation_ok else '✗'} "
              f"faith={faith_score}/5 "
              f"ground={'✓' if grounding_pass else '✗'} "
              f"r{rounds} {elapsed:.1f}s")

    # 指标一律从 results 聚合（断点续跑时计数器只覆盖新批，不能用）
    n = len(results)
    results_by_id = {r["id"]: r for r in results}
    # 引据合规分母：只统计带 golden_chapter 的题（开放性知识题无单一篇目出处，不适用此指标）
    cite_targets = [q["id"] for q in questions if q.get("golden_chapter")]
    n_citation = len(cite_targets)
    recall_hits = sum(1 for r in results if r["recall"])
    mrr_sum = sum((1.0 / r["mrr_rank"]) if r["mrr_rank"] else 0 for r in results)
    cite_ok = sum(1 for r in results if r["citation_ok"])
    faith_sum = sum(r["faithfulness"] for r in results)
    ground_pass = sum(1 for r in results if r["grounding_pass"])
    elapsed_total = sum(r["elapsed_s"] for r in results)

    # 按题型的引据合规（分子分母都用带 golden_chapter 的题）
    cite_by_type: dict[str, tuple[int, int]] = {}
    for q in questions:
        if q.get("golden_chapter"):
            t = q["type"]
            ok = 1 if results_by_id.get(q["id"], {}).get("citation_ok") else 0
            cur = cite_by_type.get(t, (0, 0))
            cite_by_type[t] = (cur[0] + ok, cur[1] + 1)

    metrics = {
        "total_questions": n,
        "recall_at_5": round(recall_hits / n, 4) if n else 0,
        "mrr": round(mrr_sum / n, 4) if n else 0,
        "citation_numerator": cite_ok,
        "citation_denominator": n_citation,
        "citation_compliance": round(cite_ok / n_citation, 4) if n_citation else 0,
        "citation_by_type": cite_by_type,
        "faithfulness_avg": round(faith_sum / n, 2) if n else 0,
        "grounding_pass_rate": round(ground_pass / n, 4) if n else 0,
        "avg_rounds": round(sum(r["rounds"] for r in results) / n, 2) if n else 0,
        "avg_elapsed_s": round(elapsed_total / n, 1) if n else 0,
        "total_elapsed_s": round(elapsed_total, 1),
    }

    return {"metrics": metrics, "results": results}


def write_report(report: dict, path: str = "eval/report.md") -> None:
    m = report["metrics"]
    lines = [
        "# 识文新裁 S3 评测报告",
        "",
        f"## 概览",
        f"- 评测题数：{m['total_questions']}",
        f"- 总耗时：{m['total_elapsed_s']}s（平均 {m['avg_elapsed_s']}s/题）",
        "",
        "## 指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| **Recall@5** | {m['recall_at_5']:.2%} | golden 书出现在 Top-5 检索结果中的比例 |",
        f"| **MRR** | {m['mrr']:.4f} | golden 书的平均倒数排名 |",
        f"| **引据合规率** | {m['citation_numerator']}/{m['citation_denominator']} = {m['citation_compliance']:.2%} | golden 书·篇同时命中；分母=带 golden_chapter 的题（开放性知识题无单一篇目出处，不计入） |",
        f"| **Faithfulness 均分** | {m['faithfulness_avg']}/5 | LLM-as-judge 忠实度评分均值 |",
        f"| **Grounding 通过率** | {m['grounding_pass_rate']:.2%} | grounding 校验通过的比例 |",
        f"| **平均检索轮数** | {m['avg_rounds']} | 多跳收敛所需平均轮数 |",
        f"| **平均延迟** | {m['avg_elapsed_s']}s | 端到端平均耗时 |",
        "",
        "### 引据合规率（按题型，分母=带 golden_chapter 的题）",
        "",
        "| 题型 | 合规 | 值 |",
        "|---|---|---|",
    ]

    for t in ["考据", "翻译", "知识"]:
        if t in m["citation_by_type"]:
            ok, tot = m["citation_by_type"][t]
            rate = f"{ok}/{tot} = {ok/tot:.0%}" if tot else "-"
            lines.append(f"| {t} | {rate} | "
                         f"{'全部合规' if ok == tot else f'失败 {tot - ok} 题'} |")

    lines += [
        "",
        "## 各题详情",
        "",
        "| ID | 类型 | Recall | MRR | 引据 | Faith | Ground | 轮数 | 延迟 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in report["results"]:
        lines.append(
            f"| {r['id']} | {r['type']} | "
            f"{'✓' if r['recall'] else '✗'} | "
            f"{1.0/r['mrr_rank'] if r['mrr_rank'] else 0:.3f} | "
            f"{'✓' if r['citation_ok'] else '✗'} | "
            f"{r['faithfulness']}/5 | "
            f"{'✓' if r['grounding_pass'] else '✗'} | "
            f"{r['rounds']} | "
            f"{r['elapsed_s']}s |"
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 评测")
    parser.add_argument("--limit", type=int, default=None, help="限制评测题数")
    parser.add_argument("--output", type=str, default="eval/report.md")
    parser.add_argument("--resume", action="store_true", help="从断点续跑（跳过已完成题）")
    args = parser.parse_args()

    print(f"[eval] 加载评测集...")
    questions = load_questions()
    print(f"[eval] 共 {len(questions)} 题，"
          f"考据={sum(1 for q in questions if q['type']=='考据')} "
          f"翻译={sum(1 for q in questions if q['type']=='翻译')} "
          f"知识={sum(1 for q in questions if q['type']=='知识')}")

    if not get_settings().deepseek_api_key:
        print("[eval] ❌ DEEPSEEK_API_KEY 未配置，无法运行评测")
        return

    print(f"[eval] 开始评测...\n")
    report = evaluate(limit=args.limit, resume=args.resume)
    write_report(report, path=args.output)

    m = report["metrics"]
    print(f"\n[eval] 完成：")
    print(f"  Recall@5       = {m['recall_at_5']:.2%}")
    print(f"  MRR            = {m['mrr']:.4f}")
    print(f"  引据合规率     = {m['citation_numerator']}/{m['citation_denominator']} = {m['citation_compliance']:.2%}"
          f"（分母=带 golden_chapter 的 {m['citation_denominator']} 题）")
    print(f"  Faithfulness   = {m['faithfulness_avg']}/5")
    print(f"  Grounding      = {m['grounding_pass_rate']:.2%}")
    print(f"  平均轮数       = {m['avg_rounds']}")
    print(f"  平均延迟       = {m['avg_elapsed_s']}s")


if __name__ == "__main__":
    main()