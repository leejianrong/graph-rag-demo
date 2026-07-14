"""Env-driven configuration (ADR: config/secrets via ``.env`` / environment).

A single :class:`Settings` (pydantic-settings) reads endpoints, index/topic/bucket
names and the log level from the environment (or a local ``.env``). No secrets are
hardcoded; defaults match the docker-compose service names later slices use
(``minio``, ``elasticsearch``, ``kafka``). Use :func:`get_settings` for a cached
accessor.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment / ``.env``.

    Env vars are the field names upper-cased (e.g. ``MINIO_ENDPOINT``); the
    ``.env`` file is read if present. See ``.env.example`` for every variable and
    its local-dev default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- MinIO / ObjectStore -------------------------------------------------
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    minio_bucket: str = "documents"

    # --- Elasticsearch / DocumentStore + EntityStore ------------------------
    elasticsearch_url: str = "http://elasticsearch:9200"
    documents_index: str = "documents"
    entities_index: str = "entities"

    # --- Kafka / messaging seam ---------------------------------------------
    kafka_bootstrap_servers: str = "kafka:9092"
    ingest_trigger_topic: str = "ingest-triggers"

    # --- NER stage (V2, ADR-0002) -------------------------------------------
    # The spaCy pipeline the real stack loads. Defaults to the transformer model
    # for accuracy; ``SpacyNerStage`` falls back to ``en_core_web_lg`` then
    # ``en_core_web_sm`` if a heavier model is not installed. The heavy trf model
    # needs the ``trf`` optional extra (spacy-transformers + torch).
    ner_model: str = "en_core_web_trf"

    # --- LLM client (V3, ADR-0008) ------------------------------------------
    # Provider-agnostic via LiteLLM: the model is a LiteLLM model string, so any
    # OpenAI-compatible endpoint (incl. DeepSeek) is swappable here. The per-stage
    # model is config: coref pins B6 (``gpt-4o-mini``); KG-build (V5) gets its own.
    coref_model: str = "gpt-4o-mini"
    kg_build_model: str = "gpt-4o-mini"
    # Persistent response cache dir (gitignored). A cache hit costs $0 / no call.
    llm_cache_dir: str = ".cache/llm"
    # Structured-output retries on parse/validation failure before raising.
    llm_max_retries: int = 2
    # API key STRICTLY from the environment — never hardcoded. Optional so import
    # and the fakes-only fast suite never need a key; only a real provider call
    # does. Read from ``OPENAI_API_KEY`` (LiteLLM's default for the gpt-4o-mini
    # default); point ``coref_model`` at another provider + set its key to swap.
    openai_api_key: str | None = None

    # --- Logging seam --------------------------------------------------------
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (constructed once per process)."""
    return Settings()
