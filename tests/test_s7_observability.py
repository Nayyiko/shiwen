"""S7 可观测性 单元测试（无 API key 可跑）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shiwen.observability import (
    normalize_trace,
    aggregate_trace,
    trace_total_ms,
    build_stage_summary,
    format_value,
)


class TestTraceNormalize:
    def test_normalize_standard(self):
        trace = [
            {"node": "retrieve", "elapsed_ms": 100},
            {"node": "generate", "elapsed_ms": 200},
        ]
        out = normalize_trace(trace)
        assert len(out) == 2
        assert out[0] == {"node": "retrieve", "elapsed_ms": 100.0}

    def test_normalize_tolerates_missing_fields(self):
        trace = [
            {"node": "retrieve", "elapsed_ms": 100},
            {"node": "generate"},  # 缺 elapsed_ms
            {"not_a_dict"},        # 非 dict
            {"node": "write", "elapsed_s": 0.5},  # 秒单位
        ]
        out = normalize_trace(trace)
        assert len(out) == 3  # 跳过 generate 和 not_a_dict? 不，generate 无 ms 也会保留
        # generate 保留但 ms=0
        assert {"node": "generate", "elapsed_ms": 0.0} in out
        assert {"node": "write", "elapsed_ms": 500.0} in out

    def test_normalize_none(self):
        assert normalize_trace(None) == []
        assert normalize_trace([]) == []


class TestTraceAggregate:
    def test_aggregate_basic(self):
        traces = [
            [{"node": "retrieve", "elapsed_ms": 100},
             {"node": "generate", "elapsed_ms": 200}],
            [{"node": "retrieve", "elapsed_ms": 300},
             {"node": "generate", "elapsed_ms": 400}],
        ]
        stats = aggregate_trace(traces)
        assert set(stats.keys()) == {"retrieve", "generate"}
        assert stats["retrieve"].count == 2
        assert stats["retrieve"].total_ms == 400.0
        assert stats["retrieve"].avg_ms == 200.0
        assert stats["retrieve"].max_ms == 300.0
        assert stats["generate"].count == 2
        assert stats["generate"].total_ms == 600.0

    def test_aggregate_empty(self):
        assert aggregate_trace([]) == {}

    def test_trace_total(self):
        trace = [
            {"node": "a", "elapsed_ms": 100},
            {"node": "b", "elapsed_ms": 250},
        ]
        assert trace_total_ms(trace) == 350.0


class TestStageSummary:
    def test_build_stage_summary(self):
        metrics = {
            "recall_at_5": 0.91,
            "mrr": 0.7207,
            "avg_elapsed_s": 9.3,
        }
        summary = build_stage_summary("S3", "检索", metrics)
        assert summary.stage == "S3"
        assert summary.name == "检索"
        assert len(summary.metrics) == 3
        # unit 推断：rate → %
        recall = [m for m in summary.metrics if m.name == "recall_at_5"][0]
        assert recall.unit == "%"
        elapsed = [m for m in summary.metrics if m.name == "avg_elapsed_s"][0]
        assert elapsed.unit == "s"

    def test_format_value(self):
        assert format_value(0.91, "%") == "91%"
        assert format_value(9.3, "s") == "9.3s"
        assert format_value(0.7207, "") == "0.721"


class TestRunAllMetrics:
    """eval/run_all.py 的各阶段指标重算函数。"""
    def test_compute_s5_metrics(self):
        from eval.run_all import compute_s5_metrics
        results = [
            {"citation_rate": 1.0, "book_coverage": 0.33, "elapsed_s": 77.4},
            {"citation_rate": 1.0, "book_coverage": 0.33, "elapsed_s": 31.9},
            {"citation_rate": 1.0, "book_coverage": 0.33, "elapsed_s": 32.7},
        ]
        m = compute_s5_metrics(results)
        assert m["citation_traceability"] == 1.0
        assert m["book_coverage"] == 0.33
        assert m["avg_elapsed_s"] == 47.3

    def test_compute_s6_metrics(self):
        from eval.run_all import compute_s6_metrics
        results = [
            {"citation_rate": 1.0, "persona_rate": 0.5, "elapsed_s": 49.9},
            {"citation_rate": 0.0, "persona_rate": 1.0, "elapsed_s": 3.3},
        ]
        m = compute_s6_metrics(results)
        assert m["citation_compliance"] == 0.5
        assert m["persona_keyword_rate"] == 0.75
        assert m["avg_elapsed_s"] == 26.6

    def test_compute_empty(self):
        from eval.run_all import compute_s3_metrics, compute_s4_metrics
        assert compute_s3_metrics([]) == {}
        assert compute_s4_metrics([]) == {}