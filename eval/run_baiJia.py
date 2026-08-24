"""S6 角色扮演 BaiJia 六维评测：DeepSeek LLM-judge 打分。

六维（BaiJia 框架）：一致性 / 对话能力 / 角色准确性 / 信息量 / 上下文相关性 / 角色魅力。
对每个角色扮演回复，用 DeepSeek 按六维打 1-5 分，汇总六维均值。

运行：
  docker compose run --rm -e HF_ENDPOINT=https://hf-mirror.com backend bash -c "python eval/run_baiJia.py"
  docker compose run --rm -e HF_ENDPOINT=https://hf-mirror.com backend bash -c "python eval/run_baiJia.py --limit 3"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from src.shiwen.agents.personas import SAGES
from src.shiwen.config import get_settings
from src.shiwen.roleplay.graph import build_roleplay_graph

CHECKPOINT = Path("eval/baiJia_results.json")

# 六维（BaiJia 框架，英文 key + 中文名）
SIX_DIMS = [
    ("consistency", "一致性"),
    ("dialogue", "对话能力"),
    ("accuracy", "角色准确性"),
    ("informativeness", "信息量"),
    ("relevance", "上下文相关性"),
    ("appeal", "角色魅力"),
]


def load_cases(path: str = "eval/roleplay.yaml") -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["roleplay_cases"]


def judge(sage_name: str, school: str, message: str, response: str,
          client: OpenAI, model: str) -> dict:
    """用 DeepSeek 对回复做六维打分，返回 {dim: score}。"""
    dims_desc = "\n".join(f"{i+1}. {zh}（{en}）" for i, (en, zh) in enumerate(SIX_DIMS))
    prompt = f"""你是古籍角色扮演评测员。请对下面这位先贤的回复，按六个维度打分（1-5 分，5 分最佳，允许 0.5）。

## 六维
{dims_desc}

## 先贤
{sage_name}（{school}）

## 用户问题
{message}

## 先贤回复
{response}

返回 JSON（不要 markdown 标记）：
{{"consistency": 分数, "dialogue": 分数, "accuracy": 分数, "informativeness": 分数, "relevance": 分数, "appeal": 分数}}"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
        obj = json.loads(m.group(0) if m else content)
        scores = {}
        for en, _ in SIX_DIMS:
            v = float(obj.get(en, 0))
            scores[en] = max(0.0, min(5.0, v))
        return scores
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {en: 0.0 for en, _ in SIX_DIMS}


def evaluate(limit: int | None = None) -> dict:
    cases = load_cases()
    if limit:
        cases = cases[:limit]

    s = get_settings()
    client = OpenAI(api_key=s.deepseek_api_key, base_url=s.deepseek_base_url)
    graph = build_roleplay_graph()

    results: list[dict] = []
    dim_sums = {en: 0.0 for en, _ in SIX_DIMS}
    total_elapsed = 0.0

    for i, c in enumerate(cases):
        cid = c["id"]
        sage_id = c["sage_id"]
        message = c["message"]
        persona = SAGES[sage_id]

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
        response = result.get("response", "")
        elapsed = time.time() - t0
        total_elapsed += elapsed

        scores = judge(persona.name, persona.school, message, response, client, s.deepseek_model)
        for en, _ in SIX_DIMS:
            dim_sums[en] += scores[en]

        avg = sum(scores.values()) / len(scores)
        print(f"  六维均值 {avg:.2f}/5  {elapsed:.1f}s")

        results.append({
            "id": cid,
            "sage_id": sage_id,
            "message": message,
            "response_preview": response[:200],
            "scores": scores,
            "avg": round(avg, 2),
            "elapsed_s": round(elapsed, 1),
        })

        CHECKPOINT.write_text(json.dumps(
            {"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(results)
    dim_avg = {en: round(dim_sums[en] / n, 2) for en, _ in SIX_DIMS}
    overall = round(sum(dim_avg.values()) / len(dim_avg), 2)

    metrics = {
        "total_cases": n,
        "dim_avg": dim_avg,
        "overall_avg": overall,
        "avg_elapsed_s": round(total_elapsed / n, 1) if n else 0,
    }
    return {"metrics": metrics, "results": results}


def write_report(report: dict, path: str = "eval/report_baiJia.md") -> None:
    m = report["metrics"]
    lines = [
        "# 识文新裁 S6 角色扮演 BaiJia 六维评测报告",
        "",
        "## 概览",
        f"- 评测用例数：{m['total_cases']}",
        f"- judge：DeepSeek LLM-judge（1-5 分，温度 0）",
        "",
        "## 六维均值",
        "",
        "| 维度 | 均值 |",
        "|---|---|",
    ]
    for en, zh in SIX_DIMS:
        lines.append(f"| {zh}（{en}） | {m['dim_avg'][en]} |")
    lines.append(f"| **总均值** | **{m['overall_avg']}** |")

    lines += [
        "",
        "## 各用例详情",
        "",
        "| ID | 先贤 | 问题 | 六维均值 |",
        "|---|---|---|---|",
    ]
    for r in report["results"]:
        lines.append(f"| {r['id']} | {r['sage_id']} | {r['message'][:20]} | {r['avg']} |")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S6 角色扮演 BaiJia 六维评测")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="eval/report_baiJia.md")
    args = parser.parse_args()

    if not get_settings().deepseek_api_key:
        print("[eval] ❌ DEEPSEEK_API_KEY 未配置")
        return

    print("[eval] 开始 BaiJia 六维评测...\n")
    report = evaluate(limit=args.limit)
    write_report(report, path=args.output)

    m = report["metrics"]
    print(f"\n[eval] 完成：")
    for en, zh in SIX_DIMS:
        print(f"  {zh}: {m['dim_avg'][en]}/5")
    print(f"  总均值: {m['overall_avg']}/5")


if __name__ == "__main__":
    main()
