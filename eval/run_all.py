"""S7 统一评测编排器：聚合四个阶段的评测结果 → 一份总报告。

不重新跑评测（避免重复 LLM 费用），而是读取各阶段的断点 checkpoint，
重新计算指标后用 observability 模块聚合为统一视图。

运行：
  python eval/run_all.py                      # 读取现有 checkpoint，产出 report_all.md
  python eval/run_all.py --output eval/report_all.md

依赖各阶段的 checkpoint（由对应 run_*.py 生成）：
  - eval/results.json        (S3 检索)
  - eval/debate_results.json (S4 辩论)
  - eval/writing_results.json (S5 写作)
  - eval/roleplay_results.json (S6 角色扮演)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shiwen.observability import (
    StageSummary,
    build_stage_summary,
    format_value,
)

ROOT = Path(__file__).resolve().parent


# ── 各阶段指标重算 ────────────────────────────────────────────────────────────


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def compute_s3_metrics(results: list[dict]) -> dict[str, float]:
    """S3 检索指标：Recall@5 / MRR / 引据合规 / Faithfulness / Grounding。

    引据合规分母 = 带 golden_chapter 的题（需 questions.yaml）。
    """
    n = len(results)
    if n == 0:
        return {}

    recall = _mean([1.0 if r.get("recall") else 0.0 for r in results])
    mrr = _mean([1.0 / r["mrr_rank"] if r.get("mrr_rank") else 0.0 for r in results])
    faith = _mean([float(r.get("faithfulness", 0)) for r in results])
    ground = _mean([1.0 if r.get("grounding_pass") else 0.0 for r in results])
    avg_elapsed = _mean([float(r.get("elapsed_s", 0)) for r in results])
    avg_rounds = _mean([float(r.get("rounds", 0)) for r in results])

    # 引据合规：需要 questions.yaml 确定分母
    qpath = ROOT / "questions.yaml"
    n_citation = 0
    cite_ok = 0
    if qpath.exists():
        import yaml
        qs = yaml.safe_load(qpath.read_text(encoding="utf-8"))["questions"]
        by_id = {r["id"]: r for r in results}
        for q in qs:
            if q.get("golden_chapter"):
                n_citation += 1
                if by_id.get(q["id"], {}).get("citation_ok"):
                    cite_ok += 1

    citation = cite_ok / n_citation if n_citation else 0.0

    return {
        "recall_at_5": round(recall, 4),
        "mrr": round(mrr, 4),
        "citation_compliance": round(citation, 4),
        "faithfulness": round(faith, 2),
        "grounding_pass_rate": round(ground, 4),
        "avg_rounds": round(avg_rounds, 2),
        "avg_elapsed_s": round(avg_elapsed, 1),
    }


def compute_s4_metrics(results: list[dict]) -> dict[str, float]:
    """S4 辩论指标：路由正确率 / 论据引用率 / 漂移。"""
    n = len(results)
    if n == 0:
        return {}

    routing = _mean([1.0 if r.get("routing_ok") else 0.0 for r in results])

    total_citations = 0
    valid_citations = 0
    total_drift = 0
    for r in results:
        for s in r.get("speeches_citation", []):
            total_citations += s.get("citations_total", 0)
            valid_citations += s.get("citations_valid", 0)
        total_drift += r.get("drift_count", 0)

    citation = valid_citations / total_citations if total_citations else 0.0
    avg_elapsed = _mean([float(r.get("elapsed_s", 0)) for r in results])

    return {
        "routing_accuracy": round(routing, 4),
        "citation_compliance": round(citation, 4),
        "total_drift_events": float(total_drift),
        "avg_elapsed_s": round(avg_elapsed, 1),
    }


def compute_s5_metrics(results: list[dict]) -> dict[str, float]:
    """S5 写作指标：引据可回溯率 / 书目覆盖率。"""
    n = len(results)
    if n == 0:
        return {}

    citation = _mean([float(r.get("citation_rate", 0)) for r in results])
    coverage = _mean([float(r.get("book_coverage", 0)) for r in results])
    avg_elapsed = _mean([float(r.get("elapsed_s", 0)) for r in results])

    return {
        "citation_traceability": round(citation, 4),
        "book_coverage": round(coverage, 4),
        "avg_elapsed_s": round(avg_elapsed, 1),
    }


def compute_s6_metrics(results: list[dict]) -> dict[str, float]:
    """S6 角色扮演指标：引据合规率 / 人设关键词命中率。"""
    n = len(results)
    if n == 0:
        return {}

    citation = _mean([float(r.get("citation_rate", 0)) for r in results])
    persona = _mean([float(r.get("persona_rate", 0)) for r in results])
    avg_elapsed = _mean([float(r.get("elapsed_s", 0)) for r in results])

    return {
        "citation_compliance": round(citation, 4),
        "persona_keyword_rate": round(persona, 4),
        "avg_elapsed_s": round(avg_elapsed, 1),
    }


# 阶段注册表：stage → (名称, checkpoint, 指标重算函数)
STAGES: list[tuple[str, str, str, callable]] = [
    ("S3", "检索问答", "results.json", compute_s3_metrics),
    ("S4", "先贤辩论", "debate_results.json", compute_s4_metrics),
    ("S5", "研究写作", "writing_results.json", compute_s5_metrics),
    ("S6", "角色扮演", "roleplay_results.json", compute_s6_metrics),
]


def load_checkpoint(path: str) -> list[dict] | None:
    """读取 checkpoint 的 results 字段；缺失或损坏返回 None。"""
    p = ROOT / path
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("results", [])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def aggregate() -> dict:
    """读取所有 checkpoint，返回 {stage, summary, missing}。"""
    summaries: list[StageSummary] = []
    missing: list[str] = []

    for stage, name, path, compute_fn in STAGES:
        results = load_checkpoint(path)
        if results is None:
            missing.append(stage)
            continue
        metrics = compute_fn(results)
        summaries.append(build_stage_summary(stage, name, metrics))

    return {"summaries": summaries, "missing": missing}


def write_report(agg: dict, path: str = "eval/report_all.md") -> None:
    """产出统一总报告。"""
    lines = [
        "# 识文新裁 统一评测总报告（S7）",
        "",
        "## 总览",
        "",
        "| 阶段 | 模块 | 核心指标 |",
        "|---|---|---|",
    ]

    for s in agg["summaries"]:
        # 每个阶段展示前两个核心指标
        core = "，".join(
            f"{m.name} = {format_value(m.value, m.unit)}"
            for m in s.metrics[:3]
        )
        lines.append(f"| {s.stage} | {s.name} | {core} |")

    if agg["missing"]:
        lines += [
            "",
            f"> ⚠️ 以下阶段尚未运行评测（缺 checkpoint）：{', '.join(agg['missing'])}",
            "> 运行对应 run_*.py 后重跑本脚本。",
        ]

    lines += [
        "",
        "## 各阶段明细",
        "",
    ]

    for s in agg["summaries"]:
        lines += [
            f"### {s.stage} — {s.name}",
            "",
            "| 指标 | 值 |",
            "|---|---|",
        ]
        for m in s.metrics:
            lines.append(f"| {m.name} | {format_value(m.value, m.unit)} |")
        lines.append("")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S7 统一评测编排器")
    parser.add_argument("--output", type=str, default="eval/report_all.md")
    args = parser.parse_args()

    print("[eval] 聚合各阶段评测结果...")
    agg = aggregate()

    for s in agg["summaries"]:
        print(f"  {s.stage} {s.name}: {len(s.metrics)} 项指标")
    if agg["missing"]:
        print(f"  ⚠️ 缺失：{', '.join(agg['missing'])}")

    write_report(agg, path=args.output)


if __name__ == "__main__":
    main()
