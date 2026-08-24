"""token 节省对照实验：开/关上下文治理的 prompt token 对比。

构造典型长对话场景（多轮历史 + 多条检索 chunk），对比：
- 关闭治理：全量历史 + 全量 chunk 的 token 数
- 开启治理：trim_history + compress_chunks 后的 token 数
节省比 = (关闭 - 开启) / 关闭

运行（纯本地，无需 API/Docker）：
  python eval/run_token_save.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shiwen.context import (
    CHUNK_TOKEN_BUDGET,
    HISTORY_TOKEN_BUDGET,
    build_prompt_budget,
    estimate_tokens,
)

CHECKPOINT = Path("eval/token_save_results.json")

# 模拟长对话历史：轮数越长越接近真实长对话场景
def _make_history(rounds: int) -> list[dict]:
    hist = []
    for i in range(rounds):
        hist.append({"role": "user", "content": f"这是第{i+1}个问题，关于先贤的学说、生平与时代背景，涉及仁义礼智、道法自然、法术势等多个话题。"})
        hist.append({"role": "assistant", "content": f"这是第{i+1}个回答，先贤引经据典，从「论语」「道德经」等原著里引用原文加以论证，阐述自己的学派立场与观点。"})
    return hist


# 模拟检索 chunk（每个约 500 字符，贴近真实古籍 chunk）
def _make_chunks(n: int) -> list[dict]:
    chunks = []
    base = "古之学者必有师。师者，所以传道受业解惑也。人非生而知之者，孰能无惑？惑而不从师，其为惑也，终不解矣。生乎吾前，其闻道也固先乎吾，吾从而师之；生乎吾后，其闻道也亦先乎吾，吾从而师之。吾师道也，夫庸知其年之先后生于吾乎？是故无贵无贱，无长无少，道之所存，师之所存也。"
    for i in range(n):
        chunks.append({"text": base, "book": "论语", "chapter": f"第{i+1}章"})
    return chunks


def evaluate() -> dict:
    # 场景：不同轮数/不同 chunk 数的长对话
    scenarios = [
        {"name": "短对话(6轮/3chunk)", "rounds": 6, "chunks": 3},
        {"name": "中对话(15轮/5chunk)", "rounds": 15, "chunks": 5},
        {"name": "长对话(30轮/10chunk)", "rounds": 30, "chunks": 10},
        {"name": "超长对话(50轮/20chunk)", "rounds": 50, "chunks": 20},
    ]

    results = []
    total_off = 0
    total_on = 0

    for sc in scenarios:
        history = _make_history(sc["rounds"])
        chunks = _make_chunks(sc["chunks"])

        # 关闭治理：全量
        off_tokens = sum(estimate_tokens(h["content"]) for h in history) + \
                     sum(estimate_tokens(c["text"]) for c in chunks)

        # 开启治理：裁剪
        budget = build_prompt_budget(history, chunks)
        on_tokens = budget["total_tokens"]

        save = (off_tokens - on_tokens) / off_tokens if off_tokens else 0.0
        total_off += off_tokens
        total_on += on_tokens

        results.append({
            "scenario": sc["name"],
            "off_tokens": off_tokens,
            "on_tokens": on_tokens,
            "saved_tokens": off_tokens - on_tokens,
            "save_ratio": round(save, 4),
        })

    overall_save = (total_off - total_on) / total_off if total_off else 0.0

    metrics = {
        "total_off_tokens": total_off,
        "total_on_tokens": total_on,
        "overall_save_ratio": round(overall_save, 4),
        "history_budget": HISTORY_TOKEN_BUDGET,
        "chunk_budget": CHUNK_TOKEN_BUDGET,
    }

    CHECKPOINT.write_text(json.dumps(
        {"metrics": metrics, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"metrics": metrics, "results": results}


def write_report(report: dict, path: str = "eval/report_token_save.md") -> None:
    m = report["metrics"]
    lines = [
        "# 识文新裁 上下文治理 token 节省对照实验",
        "",
        "## 方法",
        f"- 历史预算 {m['history_budget']} token，检索预算 {m['chunk_budget']} token",
        "- 关闭治理 = 全量历史 + 全量 chunk；开启治理 = trim_history + compress_chunks",
        "- token 估算：estimate_tokens（中文约 1 字符 1 token）",
        "",
        "## 结果",
        "",
        "| 场景 | 关闭(全量) | 开启(治理) | 节省 | 节省比 |",
        "|---|---|---|---|---|",
    ]
    for r in report["results"]:
        lines.append(
            f"| {r['scenario']} | {r['off_tokens']} | {r['on_tokens']} | "
            f"{r['saved_tokens']} | {r['save_ratio']:.0%} |"
        )
    lines.append(f"| **合计** | {m['total_off_tokens']} | {m['total_on_tokens']} | "
                 f"{m['total_off_tokens'] - m['total_on_tokens']} | **{m['overall_save_ratio']:.0%}** |")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] → {path}")


def main() -> None:
    print("[eval] 开始 token 节省对照实验...\n")
    report = evaluate()
    write_report(report)

    m = report["metrics"]
    print(f"\n[eval] 完成：")
    print(f"  关闭治理总 token: {m['total_off_tokens']}")
    print(f"  开启治理总 token: {m['total_on_tokens']}")
    print(f"  整体节省比: {m['overall_save_ratio']:.0%}")


if __name__ == "__main__":
    main()
