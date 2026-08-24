"""配置加载：从 .env / 环境变量读取，集中管理。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"

    # LLM（DeepSeek API）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Embedding（local_bge_m3 / api / cloudflare）
    embedding_provider: str = "local_bge_m3"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "BAAI/bge-m3"
    # Cloudflare Workers AI（@cf/baai/bge-m3，与 HF BAAI/bge-m3 同款 1024 维）
    cloudflare_account_id: str = ""
    cloudflare_auth_token: str = ""
    cloudflare_embedding_model: str = "@cf/baai/bge-m3"

    # PostgreSQL（docker-compose 内建）
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "shiwen"
    postgres_password: str = "change-me"
    postgres_db: str = "shiwen"

    # Milvus Lite（进程内文件模式，本地默认）
    milvus_db_path: str = "data/milvus.db"
    # 可选：覆盖为 standalone Milvus 地址（如 http://host:19530），未来换真 Milvus 只改配置不改代码
    milvus_uri: str = ""

    # Redis（会话状态 / 分层记忆 / 断点恢复）
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_enabled: bool = True  # False 时降级为进程内存储（本地无 Redis 可跑）

    # 语料下载 base URL（可选，覆盖 manifest source_base；国内可配镜像）
    corpus_raw_base: str = ""
    # HuggingFace 镜像端点（国内下载 BGE-M3 权重用，如 https://hf-mirror.com）
    hf_endpoint: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
