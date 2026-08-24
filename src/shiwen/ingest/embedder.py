"""向量化：本地 BGE-M3（sentence-transformers）或 API（OpenAI 兼容，如硅基流动）。

懒加载单例。DIM = 1024（BGE-M3 dense 维度）。

服务器资源有限（如 2G 内存）时用 API 模式（embedding_provider=api），
不吃本地内存/磁盘，query 时实时调 /embeddings 向量化。
"""

from __future__ import annotations

import math
from functools import lru_cache

import requests

from src.shiwen.config import get_settings

DIM = 1024  # BGE-M3 dense 维度


def _normalize(vec: list[float]) -> list[float]:
    """L2 归一化，配合 Milvus COSINE 检索。"""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class Embedder:
    """本地 BGE-M3（sentence-transformers，CPU）。"""

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

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "未安装 sentence-transformers。服务器请改用 EMBEDDING_PROVIDER=cloudflare（走 API，"
                    "不占本地内存）；本地开发请 pip install -r requirements-local.txt。"
                ) from e

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


class APIEmbedder:
    """API 向量化（OpenAI 兼容 /embeddings，如硅基流动 BAAI/bge-m3）。"""

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        s = get_settings()
        self.model_name = model_name or s.embedding_model
        self.api_key = api_key or s.embedding_api_key
        self.base_url = base_url or s.embedding_base_url
        self._client = None

    @property
    def dim(self) -> int:
        return DIM

    def _load(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "EMBEDDING_API_KEY 未配置。请设置硅基流动（SiliconFlow）的 embedding key。"
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def encode(self, texts: list[str], batch_size: int = 32,
               normalize: bool = True) -> list[list[float]]:
        """批量编码：调 /embeddings，逐 batch 请求（API 有单次输入长度上限）。"""
        client = self._load()
        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(model=self.model_name, input=batch)
            vecs.extend([d.embedding for d in resp.data])
        if normalize:
            vecs = [_normalize(v) for v in vecs]
        return vecs


class CloudflareEmbedder:
    """Cloudflare Workers AI 向量化（@cf/baai/bge-m3，非 OpenAI 兼容接口）。

    接口：POST /client/v4/accounts/{id}/ai/run/@cf/baai/bge-m3
          body {"text": [...]}  → 返回 {"result": {"data": [[...], ...]}}
    """

    def __init__(self, account_id: str | None = None, auth_token: str | None = None,
                 model_name: str | None = None):
        s = get_settings()
        self.account_id = account_id or s.cloudflare_account_id
        self.auth_token = auth_token or s.cloudflare_auth_token
        self.model_name = model_name or s.cloudflare_embedding_model

    @property
    def dim(self) -> int:
        return DIM

    def encode(self, texts: list[str], batch_size: int = 32,
               normalize: bool = True) -> list[list[float]]:
        """批量编码：逐 batch POST Cloudflare AI run，返回 result.data。"""
        if not self.auth_token:
            raise ValueError("CLOUDFLARE_AUTH_TOKEN 未配置")
        if not self.account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID 未配置")

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/run/{self.model_name}"
        )
        headers = {"Authorization": f"Bearer {self.auth_token}"}

        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = requests.post(url, headers=headers, json={"text": batch}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success"):
                raise RuntimeError(f"Cloudflare AI 调用失败: {data.get('errors')}")
            vecs.extend(data["result"]["data"])
        if normalize:
            vecs = [_normalize(v) for v in vecs]
        return vecs


@lru_cache
def get_embedder():
    """按 embedding_provider 返回本地 / OpenAI 兼容 API / Cloudflare 向量化器。"""
    provider = get_settings().embedding_provider
    if provider == "cloudflare":
        return CloudflareEmbedder()
    if provider == "api":
        return APIEmbedder()
    return Embedder()
