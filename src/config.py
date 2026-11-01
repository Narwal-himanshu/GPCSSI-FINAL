"""Application settings, loaded from environment variables (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- General -------------------------------------------------------------
    log_level: str = "INFO"
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    # Comma-separated list of allowed CORS origins. Empty = same-origin only.
    cors_origins: str = ""

    # --- Upload limits -------------------------------------------------------
    max_upload_mb: float = 50.0
    max_lines: int = 200_000
    # Maximum number of analyses that may run at the same time.
    max_concurrent_jobs: int = 2

    # --- Ollama --------------------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: float = 120.0
    # Comma-separated list of Ollama models offered in the UI.
    ollama_models: str = "llama3.2:1b,gemma4:12b,qwen2.5:0.5b"
    default_model: str = "llama3.2:1b"

    # --- Gemini --------------------------------------------------------------
    # Optional server-side key, used when the browser does not supply one.
    google_api_key: str | None = None
    gemini_max_rpm: float = 15.0
    gemini_burst: int = 1
    gemini_cooldown_seconds: float = 15.0
    gemini_max_retries: int = 5
    gemini_base_delay: float = 2.0
    gemini_max_delay: float = 30.0
    gemini_cache_ttl: int = 300
    gemini_timeout: float = 60.0
    gemini_max_concurrency: int = 2

    # --- ML pre-filter -------------------------------------------------------
    prefilter_min_lines: int = 50
    prefilter_percentile: float = 85.0
    prefilter_map_size: int = 12
    prefilter_seed: int | None = 42

    # --- Misc ----------------------------------------------------------------
    enable_gpu_status: bool = True

    @field_validator("google_api_key", "prefilter_seed", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        return None if v == "" else v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def ollama_model_list(self) -> list[str]:
        return [m.strip() for m in self.ollama_models.split(",") if m.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
