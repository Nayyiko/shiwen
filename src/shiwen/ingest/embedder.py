"""本地 BGE-M3 向量化（sentence-transformers，CPU）。

懒加载单例：权重约 2GB，仅首次调用时下载/加载，进程内复用。
"""

from __future__ import annotations

from functools import lru_cache

from src.shiwen.config import get_settings

DIM = 1024  # BGE-M3 dense 维度


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        s = get_settings()
        self.model_name = model_name or s.embedding_model
        self.device = device or "cpu"
        self._model = None

    @property
    def dim(self) -> int:
        return DIM

    def _load(self):
        if self._model is None:
            import os

            if get_settings().hf_endpoint:  # 国内下载权重可走 hf-mirror
                os.environ.setdefault("HF_ENDPOINT", get_settings().hf_endpoint)

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32,
               normalize: bool = True) -> list[list[float]]:
        """批量编码为 dense 向量（默认归一化，配合 COSINE 检索）。"""
        model = self._load()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
