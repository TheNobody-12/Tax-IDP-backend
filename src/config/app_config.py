"""Application configuration loading and validation."""
from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache

@dataclass(frozen=True)
class AppConfig:
    azure_storage_conn: str | None
    bronze_container: str
    silver_container: str
    llm_max_concurrency: int
    di_timeout: float
    llm_page_timeout: float
    environment: str
    allow_any_category_change: bool

    @property
    def have_blob(self) -> bool:
        return bool(self.azure_storage_conn)

    @property
    def is_prod(self) -> bool:
        return self.environment.lower() in {"prod","production"}


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    cfg = AppConfig(
        azure_storage_conn=os.getenv("AzureWebJobsStorage") or os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        bronze_container=os.getenv("AZURE_BRONZE_CONTAINER", "bronze"),
        silver_container=os.getenv("AZURE_SILVER_CONTAINER", "silver"),
        llm_max_concurrency=_read_env_int("LLM_MAX_CONCURRENCY", 3),
        di_timeout=_read_env_float("DI_TIMEOUT", 60.0),
        llm_page_timeout=_read_env_float("LLM_PAGE_TIMEOUT", 45.0),
        environment=os.getenv("ENV", os.getenv("ENVIRONMENT","dev")),
        allow_any_category_change=os.getenv("ALLOW_ANY_CATEGORY_CHANGE","false").lower() == "true",
    )
    # Fail-fast validation for production
    if cfg.is_prod:
        missing: list[str] = []
        if not cfg.azure_storage_conn:
            missing.append("Azure storage connection string (AzureWebJobsStorage / AZURE_STORAGE_CONNECTION_STRING)")
        if missing:
            raise RuntimeError(f"Missing required production configuration: {', '.join(missing)}")
    return cfg
