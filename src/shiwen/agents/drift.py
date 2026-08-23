"""人格漂移软监测：滑动窗口余弦相似度 + RAG 长时记忆纠偏。

简历第 2 条核心：监测纠偏，不硬约束——不禁止发言，仅在漂移时注入提醒。
"""

from __future__ import annotations

import numpy as np


class DriftMonitor:
    """滑动窗口余弦相似度人格漂移监测器。

    每位先贤独立维护一个发言嵌入的滑动窗口，新发言与窗口均值向量比较。
    低于阈值 → drift_warning，下一轮注入纠偏提示（RAG 长时记忆中的经典语录）。
    """

    def __init__(self, window_size: int = 5, threshold: float = 0.55):
        """
        Args:
            window_size: 滑动窗口大小（最近 N 条发言）
            threshold: 余弦相似度阈值，低于此值视为漂移
        """
        self.window_size = window_size
        self.threshold = threshold
        self._windows: dict[str, list[np.ndarray]] = {}  # sage_id → 最近 N 条发言的 embedding
        self._drift_log: list[DriftEvent] = []

    def observe(self, sage_id: str, speech: str,
                embedder) -> DriftEvent | None:
        """观察一条发言，检测是否漂移。

        Returns:
            DriftEvent 若漂移，否则 None。
        """
        vec = embedder.encode([speech])[0]
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        window = self._windows.setdefault(sage_id, [])
        window.append(vec)
        if len(window) > self.window_size:
            window.pop(0)

        if len(window) < 2:
            return None  # 发言不足窗口大小，不检测

        # 窗口均值向量（排除当前发言以避免自相关）
        mean_vec = np.mean(window[:-1], axis=0)
        mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)

        similarity = float(np.dot(vec, mean_vec))

        if similarity < self.threshold:
            event = DriftEvent(
                sage_id=sage_id,
                round_index=len(window) - 1,
                similarity=round(similarity, 4),
                threshold=self.threshold,
            )
            self._drift_log.append(event)
            return event

        return None

    def get_correction_hint(self, sage_id: str, sage_name: str,
                            anchor_chunks: list[str]) -> str:
        """生成纠偏提示（注入下一轮发言 prompt）。

        anchor_chunks: RAG 检索的该先贤经典语录（用于长时记忆修正）。
        """
        if not anchor_chunks:
            return ""

        chunks_text = "\n".join(
            f"  · {c[:120]}" for c in anchor_chunks[:3]
        )
        return (
            f"\n\n## ⚠️ 风格提醒\n"
            f"你最近的发言有偏离你一贯立论风格的趋势。请回顾以下你本人的经典论述，"
            f"在接下来的发言中回归你的本色：\n"
            f"{chunks_text}\n"
            f"注意：这只是一个提醒，不是限制你的表达——你仍然可以自由阐述观点，"
            f"但请确保你的主张与你的学派立场一致。"
        )

    def get_drift_log(self) -> list[dict]:
        """返回漂移日志（用于评测/metrics）。"""
        return [e.to_dict() for e in self._drift_log]

    def reset(self) -> None:
        self._windows.clear()
        self._drift_log.clear()


class DriftEvent:
    """一次漂移事件记录。"""
    def __init__(self, sage_id: str, round_index: int,
                 similarity: float, threshold: float):
        self.sage_id = sage_id
        self.round_index = round_index
        self.similarity = similarity
        self.threshold = threshold

    def to_dict(self) -> dict:
        return {
            "sage_id": self.sage_id,
            "round_index": self.round_index,
            "similarity": self.similarity,
            "threshold": self.threshold,
        }

    def __repr__(self) -> str:
        return (f"DriftEvent(sage={self.sage_id}, round={self.round_index}, "
                f"sim={self.similarity:.3f}<{self.threshold})")