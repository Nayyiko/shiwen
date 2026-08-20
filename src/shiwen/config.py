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

    # Embedding（local_bge_m3 / api）
    embedding_provider: str = "local_bge_m3"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "BAAI/bge-m3"

    # PostgreSQL（docker-compose 内建）
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "shiwen"
    postgres_password: str = "change-me"
    postgres_db: str = "shiwen"

    # Milvus Lite（进程内文件模式）
    milvus_db_path: str = "data/milvus.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
