from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "QuestMate"
    app_env: str = "development"
    log_level: str = "info"
    knowledge_admin_token: str = ""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    custom_model_endpoint_hosts: str = ""

    tavily_api_key: str = ""

    database_url: str = "postgresql+asyncpg://questmate:questmate@localhost:5432/questmate"
    redis_url: str = "redis://localhost:6379/0"
    database_pool_size: int = Field(default=10, ge=1, le=50)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_pool_timeout_seconds: int = Field(default=30, ge=1, le=120)
    allow_in_memory_storage: bool = False
    cors_allowed_origins: str = "http://localhost:1420,http://127.0.0.1:1420,tauri://localhost"

    search_max_results: int = Field(default=5, ge=1, le=20)
    search_include_raw_content: bool = True
    evidence_passage_max_chars: int = Field(default=1600, ge=400, le=6000)
    external_request_timeout_seconds: int = Field(default=20, ge=3, le=90)
    # Identity discovery is a guard before the main answer path. Keep each
    # attempt short and bounded so a slow upstream cannot consume the entire
    # request budget before the user can choose a candidate.
    identity_resolution_timeout_seconds: int = Field(default=8, ge=3, le=30)
    identity_resolution_attempts: int = Field(default=2, ge=1, le=2)
    # Must remain below evaluator/client deadlines so provider failures can
    # degrade to a conservative response instead of becoming an API timeout.
    model_request_timeout_seconds: int = Field(default=40, ge=5, le=55)
    tavily_max_concurrency: int = Field(default=3, ge=1, le=8)
    tavily_search_cache_ttl_seconds: int = Field(default=86400, ge=0, le=2592000)
    tavily_search_cache_max_entries: int = Field(default=512, ge=16, le=10000)
    search_cache_use_redis: bool = True
    tavily_first_wave_queries: int = Field(default=2, ge=1, le=4)
    tavily_max_queries_per_request: int = Field(default=2, ge=1, le=8)
    mediawiki_direct_search: bool = True
    wiki_auto_index_enabled: bool = True
    wiki_auto_index_pages_per_query: int = Field(default=3, ge=1, le=10)
    wiki_link_expansion_pages_per_query: int = Field(default=2, ge=0, le=6)
    knowledge_auto_index_ttl_seconds: int = Field(default=604800, ge=0, le=7776000)
    knowledge_retrieval_results: int = Field(default=4, ge=1, le=12)
    # Evaluation-only retrieval hints are untrusted request metadata.  They are
    # disabled unless an isolated evaluator opts in explicitly, and are never
    # accepted by production deployments.
    allow_evaluation_retrieval_hints: bool = False
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=64, le=4096)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @property
    def allowed_custom_model_hosts(self) -> set[str]:
        return {
            host.casefold().strip().strip(".")
            for host in self.custom_model_endpoint_hosts.split(",")
            if host.strip()
        }

    @property
    def is_production(self) -> bool:
        """Fail closed unless the environment is explicitly local or test."""
        return self.app_env.strip().casefold() not in {
            "development",
            "dev",
            "local",
            "test",
            "testing",
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
