"""向量化器 单元测试（无 API key / 无模型可跑，全 mock）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.shiwen.ingest.embedder import (
    DIM,
    APIEmbedder,
    CloudflareEmbedder,
    Embedder,
    _normalize,
)


class TestNormalize:
    def test_normalize_unit(self):
        v = _normalize([3.0, 4.0])
        assert abs(v[0] - 0.6) < 1e-9
        assert abs(v[1] - 0.8) < 1e-9

    def test_normalize_zero_vector(self):
        v = _normalize([0.0, 0.0])
        assert v == [0.0, 0.0]

    def test_dim(self):
        assert DIM == 1024


class TestAPIEmbedder:
    def test_encode(self):
        emb = APIEmbedder(model_name="BAAI/bge-m3", api_key="test",
                          base_url="https://api.siliconflow.cn/v1")
        mock_client = MagicMock()
        resp = MagicMock()
        resp.data = [
            MagicMock(embedding=[1.0, 0.0, 0.0]),
            MagicMock(embedding=[0.0, 1.0, 0.0]),
        ]
        mock_client.embeddings.create.return_value = resp
        emb._client = mock_client

        vecs = emb.encode(["a", "b"], normalize=False)
        assert len(vecs) == 2
        assert vecs[0] == [1.0, 0.0, 0.0]
        mock_client.embeddings.create.assert_called_once()

    def test_encode_normalize(self):
        emb = APIEmbedder(model_name="x", api_key="test", base_url="https://x")
        mock_client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[3.0, 4.0])]
        mock_client.embeddings.create.return_value = resp
        emb._client = mock_client

        vecs = emb.encode(["a"], normalize=True)
        assert abs(vecs[0][0] - 0.6) < 1e-9
        assert abs(vecs[0][1] - 0.8) < 1e-9

    def test_batching(self):
        emb = APIEmbedder(model_name="x", api_key="test", base_url="https://x")
        mock_client = MagicMock()
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[i, i]) for i in range(2)]
        mock_client.embeddings.create.return_value = resp
        emb._client = mock_client

        emb.encode(["a", "b", "c", "d"], batch_size=2, normalize=False)
        assert mock_client.embeddings.create.call_count == 2

    def test_missing_key_raises(self):
        emb = APIEmbedder(model_name="x", api_key="test", base_url="https://x")
        emb.api_key = ""  # 直接覆盖，避免读到 .env 真实 key
        try:
            emb._load()
            assert False, "应抛 ValueError"
        except ValueError as e:
            assert "EMBEDDING_API_KEY" in str(e)


class TestCloudflareEmbedder:
    def test_encode(self):
        emb = CloudflareEmbedder(
            account_id="7465b2132610a9efb2acae1c1298b089",
            auth_token="test-token",
            model_name="@cf/baai/bge-m3",
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": {"data": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
        }
        mock_resp.raise_for_status.return_value = None

        with patch("src.shiwen.ingest.embedder.requests.post", return_value=mock_resp) as mp:
            vecs = emb.encode(["a", "b"], normalize=False)

        assert len(vecs) == 2
        assert vecs[0] == [1.0, 0.0, 0.0]
        # 验证 URL + header + body
        args, kwargs = mp.call_args
        url = args[0]
        assert "api.cloudflare.com/client/v4/accounts/7465b2132610a9efb2acae1c1298b089" in url
        assert "@cf/baai/bge-m3" in url
        assert kwargs["json"] == {"text": ["a", "b"]}
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"

    def test_encode_normalize(self):
        emb = CloudflareEmbedder(account_id="x", auth_token="t", model_name="m")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "result": {"data": [[3.0, 4.0]]},
        }
        with patch("src.shiwen.ingest.embedder.requests.post", return_value=mock_resp):
            vecs = emb.encode(["a"], normalize=True)
        assert abs(vecs[0][0] - 0.6) < 1e-9
        assert abs(vecs[0][1] - 0.8) < 1e-9

    def test_missing_token_raises(self):
        emb = CloudflareEmbedder(account_id="x", auth_token="test", model_name="m")
        emb.auth_token = ""  # 直接覆盖，避免读到 .env 真实 token
        try:
            emb.encode(["a"])
            assert False, "应抛 ValueError"
        except ValueError as e:
            assert "CLOUDFLARE_AUTH_TOKEN" in str(e)

    def test_api_failure_raises(self):
        emb = CloudflareEmbedder(account_id="x", auth_token="t", model_name="m")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": False, "errors": ["boom"]}
        with patch("src.shiwen.ingest.embedder.requests.post", return_value=mock_resp):
            try:
                emb.encode(["a"])
                assert False, "应抛 RuntimeError"
            except RuntimeError as e:
                assert "失败" in str(e)


class TestGetEmbedder:
    def _mock_settings(self, **kw):
        defaults = dict(
            embedding_provider="local_bge_m3",
            embedding_api_key="",
            embedding_base_url="",
            embedding_model="BAAI/bge-m3",
            cloudflare_account_id="",
            cloudflare_auth_token="",
            cloudflare_embedding_model="@cf/baai/bge-m3",
            hf_endpoint="",
        )
        defaults.update(kw)
        return SimpleNamespace(**defaults)

    def test_api_provider(self):
        from src.shiwen.ingest import embedder
        embedder.get_embedder.cache_clear()
        try:
            with patch('src.shiwen.ingest.embedder.get_settings') as m:
                m.return_value = self._mock_settings(embedding_provider="api", embedding_api_key="k")
                assert isinstance(embedder.get_embedder(), APIEmbedder)
        finally:
            embedder.get_embedder.cache_clear()

    def test_cloudflare_provider(self):
        from src.shiwen.ingest import embedder
        embedder.get_embedder.cache_clear()
        try:
            with patch('src.shiwen.ingest.embedder.get_settings') as m:
                m.return_value = self._mock_settings(embedding_provider="cloudflare")
                assert isinstance(embedder.get_embedder(), CloudflareEmbedder)
        finally:
            embedder.get_embedder.cache_clear()

    def test_local_provider_default(self):
        from src.shiwen.ingest import embedder
        embedder.get_embedder.cache_clear()
        try:
            with patch('src.shiwen.ingest.embedder.get_settings') as m:
                m.return_value = self._mock_settings()
                assert isinstance(embedder.get_embedder(), Embedder)
        finally:
            embedder.get_embedder.cache_clear()