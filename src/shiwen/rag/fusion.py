"""RRF（Reciprocal Rank Fusion）融合：无权重、无超参，先跑通再上精排。

两路检索结果（向量 + BM25）按排名倒数加权合并，k=60（经典值）。
同一 chunk 在双路中 rank 越高，融合后越靠前。
"""

from __future__ import annotations


def rrf_fuse(vector_hits: list[dict], bm25_hits: list[dict],
             top_k: int = 5, k: int = 60) -> list[dict]:
    """RRF 融合两路检索结果，返回 top_k 个合并排序后的 chunk。

    参数：
        vector_hits: 向量检索结果（需含 id/score 及完整元数据）
        bm25_hits:   BM25 检索结果
        top_k:       最终返回数量
        k:           RRF 平滑常数（经典值 60）

    返回：
        按 RRF 分数降序的 top_k 个 chunk，每个含 rrf_score 字段
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, hit in enumerate(vector_hits):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = hit

    for rank, hit in enumerate(bm25_hits):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = hit

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    results: list[dict] = []
    for cid, rrf_score in ranked[:top_k]:
        entry = dict(chunk_map[cid])
        entry["rrf_score"] = round(rrf_score, 6)
        results.append(entry)
    return results