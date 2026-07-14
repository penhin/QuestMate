from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "QuestMate"
    app_env: str = "development"
    log_level: str = "info"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    tavily_api_key: str = ""

    database_url: str = "postgresql+asyncpg://questmate:questmate@localhost:5432/questmate"
    sync_database_url: str = "postgresql+psycopg://questmate:questmate@localhost:5432/questmate"
    redis_url: str = "redis://localhost:6379/0"
    database_pool_size: int = Field(default=10, ge=1, le=50)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_pool_timeout_seconds: int = Field(default=30, ge=1, le=120)
    allow_in_memory_storage: bool = False
    cors_allowed_origins: str = "http://localhost:1420,http://127.0.0.1:1420,tauri://localhost"

    search_max_results: int = Field(default=5, ge=1, le=20)
    external_request_timeout_seconds: int = Field(default=20, ge=3, le=90)
    tavily_max_concurrency: int = Field(default=3, ge=1, le=8)
    cache_ttl_seconds: int = Field(default=604800, ge=60)
    knowledge_retrieval_results: int = Field(default=4, ge=1, le=12)
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=64, le=4096)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
