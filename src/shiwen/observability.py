"""S7 可观测性：trace 聚合 + 指标汇总。

各业务图（检索/辩论/写作/角色扮演）已在每个节点 emit `trace` 条目
（{node, elapsed_ms}）。本模块提供统一的聚合与汇总能力，供：
- 评测脚本聚合多次运行结果
- 前端/运维查看端到端延迟分布

确定性纯函数，无副作用，可单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Trace 聚合 ────────────────────────────────────────────────────────────────


@dataclass
class NodeStats:
    """单个节点的延迟统计。"""
    node: str
    count: int = 0
    total_ms: float = 0.0
    avg_ms: float = 0.0
    max_ms: float = 0.0

    @property
    def pct(self) -> float:
        """该节点占总时长的比例（由调用方填充 total，此处返回自身 total）。"""
        return self.total_ms


def normalize_trace(trace: list[dict]) -> list[dict]:
    """规范化 trace 条目，容忍缺失字段。

    输入可能来自不同图，字段名略有差异（elapsed_ms / elapsed_s / duration）。
    统一为 {"node", "elapsed_ms"}。
    """
    out: list[dict] = []
    for t in trace or []:
        if not isinstance(t, dict) or "node" not in t:
            continue
        ms = t.get("elapsed_ms", t.get("elapsed_s", 0) * 1000)
        try:
            ms = float(ms)
        except (TypeError, ValueError):
            ms = 0.0
        out.append({"node": t["node"], "elapsed_ms": ms})
    return out


def aggregate_trace(traces: list[list[dict]]) -> dict[str, NodeStats]:
    """聚合多份 trace：按 node 统计 count/total/avg/max。

    Args:
        traces: 多次运行的 trace 列表，每次 trace 是 [{node, elapsed_ms}, ...]

    Returns:
        {node_name: NodeStats}
    """
    stats: dict[str, NodeStats] = {}
    for trace in traces:
        for entry in normalize_trace(trace):
            node = entry["node"]
            ms = entry["elapsed_ms"]
            st = stats.setdefault(node, NodeStats(node=node))
            st.count += 1
            st.total_ms += ms
            st.max_ms = max(st.max_ms, ms)

    for st in stats.values():
        if st.count:
            st.avg_ms = st.total_ms / st.count
    return stats


def trace_total_ms(trace: list[dict]) -> float:
    """单次 trace 的总耗时（毫秒）。"""
    return sum(e["elapsed_ms"] for e in normalize_trace(trace))


# ── 指标汇总 ─────────────────────────────────────────────────────────────────


@dataclass
class MetricPoint:
    """一个评测指标的数值 + 说明。"""
    name: str
    value: float
    unit: str = ""        # "%" / "s" / "" 等
    description: str = ""


@dataclass
class StageSummary:
    """单个评测阶段的汇总结果。"""
    stage: str            # S3 / S4 / S5 / S6
    name: str             # 中文名：检索 / 辩论 / 写作 / 角色扮演
    metrics: list[MetricPoint] = field(default_factory=list)
    # 阶段级延迟统计（可选）
    node_stats: dict[str, NodeStats] = field(default_factory=dict)


def build_stage_summary(
    stage: str,
    name: str,
    metrics: dict[str, float],
    traces: list[list[dict]] | None = None,
) -> StageSummary:
    """从 metrics dict 构建 StageSummary，可选附加 trace 聚合。"""
    points: list[MetricPoint] = []
    _PCT_KEYS = ("rate", "coverage", "compliance", "keyword",
                 "recall", "accuracy", "pass", "traceability")
    for k, v in metrics.items():
        unit = "%" if any(kw in k for kw in _PCT_KEYS) else ""
        if "elapsed" in k or "latency" in k:
            unit = "s"
        points.append(MetricPoint(name=k, value=v, unit=unit))

    node_stats: dict[str, NodeStats] = {}
    if traces:
        node_stats = aggregate_trace(traces)

    return StageSummary(stage=stage, name=name, metrics=points, node_stats=node_stats)


def format_value(v: float, unit: str) -> str:
    """按单位格式化指标值。"""
    if unit == "%":
        return f"{v:.0%}"
    if unit == "s":
        return f"{v:.1f}s"
    return f"{v:.3f}"
