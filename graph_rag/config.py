"""Env-driven configuration (ADR: config/secrets via ``.env`` / environment).

A single :class:`Settings` (pydantic-settings) reads endpoints, index/topic/bucket
names and the log level from the environment (or a local ``.env``). No secrets are
hardcoded; defaults match the docker-compose service names later slices use
(``minio``, ``elasticsearch``, ``kafka``). Use :func:`get_settings` for a cached
accessor.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

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
    # The most text one /v1/remember call may carry (413 above it); ADR-0014.
    max_remember_bytes: int = 1_000_000
    # The most one /ingest upload may carry (413 above it; GRA-135). Larger than remember's cap
    # because files hold markup; a deployment sets it from what its LLM budget can take.
    max_upload_bytes: int = 10_000_000
    # Requests one API key may make a minute (a burst of this many, then it refills), and failed
    # requests one address may make a minute, after which its bad keys get 429 rather than 401
    # (GRA-137). 0 turns one off. Neither applies with ``require_auth`` off.
    rate_limit_per_minute: int = 600
    failed_key_limit_per_minute: int = 20

    # --- API keys (GRA-22, GRA-23) ------------------------------------------
    # With ``require_auth`` on, every endpoint but /health needs a bearer key from this file
    # (``graph-rag-keys`` writes it). Off, requests run in the default tenant, as before.
    require_auth: bool = False
    # Where the service listens, and the opt-out for serving without auth beyond loopback
    # (graph_rag/bind.py, GRA-136). The container binds all addresses, so it needs one or the other.
    bind_host: str = "0.0.0.0"  # noqa: S104 - the container must; bind.py guards it
    allow_open_bind: bool = False
    # With auth on, the service also refuses the local-dev store credentials (GRA-138); this lets a
    # staging run use them anyway.
    allow_default_credentials: bool = False
    api_keys_path: str = ".data/api-keys.json"
    # Key management over /v1/keys (GRA-143): active keys one tenant may hold, and keys it may
    # create a minute (0 turns that limit off).
    max_keys_per_tenant: int = 10
    key_create_limit_per_minute: int = 5

    # --- Elasticsearch / DocumentStore + EntityStore ------------------------
    elasticsearch_url: str = "http://elasticsearch:9200"
    # Set a password and the client signs in as ``elasticsearch_user`` (the hosted compose
    # file turns security on and requires one, GRA-138); empty is the open local-dev default.
    elasticsearch_user: str = "elastic"
    elasticsearch_password: str = ""
    documents_index: str = "documents"
    entities_index: str = "entities"

    # --- Trigger backend / messaging seam -----------------------------------
    # ``redis`` (the default; Valkey or Redis, Streams) or ``kafka``. Each has its own settings;
    # ``INGEST_PARTITIONS`` (compose) applies to kafka only (ADR-0016).
    trigger_backend: Literal["redis", "kafka"] = "redis"
    redis_url: str = "redis://valkey:6379/0"
    # Empty is the open local-dev default; the hosted compose file requires one (GRA-146).
    redis_password: str = ""
    redis_stream: str = "ingest-triggers"
    redis_group: str = "graph-rag-ingest"
    # A trigger a worker took but never finished (it died) is handed to another after this long.
    # Generous: one document can take minutes. A running worker renews its claim, so a slow
    # document is not taken from it.
    redis_reclaim_idle_seconds: int = 900
    # The per-document lock; renewed while the handler runs, so it only needs to outlive a dead
    # worker. Keep it below the reclaim time so a reclaimed trigger finds the lock free.
    redis_lock_seconds: int = 300
    kafka_bootstrap_servers: str = "kafka:9092"
    ingest_trigger_topic: str = "ingest-triggers"
    # Where the Kafka consumer runs (GRA-31). ``inline``: a thread in the API process (the
    # default; job status is held in memory). ``publish``: the API only publishes, and a
    # separate ``graph-rag-worker`` process consumes. Both then keep job status in Elasticsearch
    # (``jobs_index``) so the API can read what the worker writes.
    ingest_mode: Literal["inline", "publish"] = "inline"
    jobs_index: str = "jobs"
    worker_health_port: int = 8001

    # --- NER stage (V2, ADR-0002) -------------------------------------------
    # The spaCy pipeline the real stack loads. Defaults to the transformer model
    # for accuracy; ``SpacyNerStage`` falls back to ``en_core_web_lg`` then
    # ``en_core_web_sm`` if a heavier model is not installed. The heavy trf model
    # needs the ``trf`` optional extra (spacy-transformers + torch).
    ner_model: str = "en_core_web_trf"

    # --- Embedder + entity linking (V4, ADR-0004/0005) ----------------------
    # Local sentence-transformer for EL + query anchoring. B1: the model and its
    # dimension are hard to change after ingestion (they pin the ES
    # ``dense_vector`` mapping), so they are fixed here. ``bge-small-en-v1.5`` →
    # 384-dim.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    # B2: the cosine merge-vs-create-new threshold for entity linking. A blocked
    # candidate scoring at or above this merges into the existing canonical
    # entity; otherwise a new canonical entity is created (ADR-0004). Fixed for
    # benchmark reproducibility (ADR-0009), but tunable via ``EL_THRESHOLD``.
    el_threshold: float = 0.82
    # How many nearest entity vectors to pull as extra EL candidates alongside the
    # type + normalized-name blocking set (ADR-0004).
    el_knn_top_k: int = 5
    # Credit-conserving EL refinements, wired but GATED OFF by default (ADR-0004):
    # an LLM tie-breaker for near-threshold decisions and NIL retention for
    # very-low-confidence entities. Off → the EL default path is deterministic and
    # $0 (no LLM call). Flip via ``EL_TIEBREAKER_ENABLED`` / ``EL_NIL_ENABLED``.
    el_tiebreaker_enabled: bool = False
    el_nil_enabled: bool = False

    # --- LLM client (V3, ADR-0008) ------------------------------------------
    # Provider-agnostic via LiteLLM: the model is a LiteLLM model string, so any
    # OpenAI-compatible endpoint (incl. DeepSeek) is swappable here. The per-stage
    # model is config: coref pins B6 (``gpt-4o-mini``); KG-build (V5) gets its own.
    coref_model: str = "gpt-4o-mini"
    kg_build_model: str = "gpt-4o-mini"
    # V7 gated prose synthesis (N17, ADR-0009, ARCHITECTURE §6). A FULLER model is
    # reserved for synthesis than for the extraction stages: coref/KG-build only
    # need cheap structured extraction, whereas turning the retrieved evidence into
    # faithful, grounded prose benefits from a stronger model. Pins B6 for the
    # synthesis stage; swappable to any LiteLLM model string via ``SYNTHESIS_MODEL``
    # (e.g. drop to ``gpt-4o-mini`` to keep it cheap). Only reached when a request
    # sets ``synthesize=true`` — the default path never builds/calls this model.
    synthesis_model: str = "gpt-4o"
    # Persistent response cache dir (gitignored). A cache hit costs $0 / no call.
    llm_cache_dir: str = ".cache/llm"
    # Structured-output retries on parse/validation failure before raising.
    llm_max_retries: int = 2
    # API key STRICTLY from the environment — never hardcoded. Optional so import
    # and the fakes-only fast suite never need a key; only a real provider call
    # does. Read from ``OPENAI_API_KEY`` (LiteLLM's default for the gpt-4o-mini
    # default); point ``coref_model`` at another provider + set its key to swap.
    openai_api_key: str | None = None
    # Bring-your-own provider key (GRA-25). On, a named tenant's LLM calls use the key it stored
    # with ``graph-rag-keys llm-key set`` (never ``openai_api_key``) and ingest for a tenant with
    # none is refused; the default tenant is unchanged. The stored keys are encrypted under
    # ``tenant_llm_key_secret``, a Fernet key, which is required when this is on.
    require_tenant_llm_key: bool = False
    tenant_llm_keys_path: str = ".data/tenant-llm-keys.json"
    tenant_llm_key_secret: str = ""

    # --- Usage metering (GRA-58) ---------------------------------------------
    # On, documents accepted, queries answered and LLM tokens are counted per tenant in
    # ``usage_path`` (one file, one service instance); ``graph-rag-admin usage`` reads it.
    meter_usage: bool = False
    usage_path: str = ".data/usage.json"

    # --- Neo4j / GraphStore (V5, ADR-0006) ----------------------------------
    # The knowledge-graph store. Defaults target the docker-compose ``neo4j``
    # service (bolt on 7687); tests/local runs point ``NEO4J_URI`` at
    # ``bolt://localhost:7687``. The password is a LOCAL-DEV default only — never
    # a real secret; override via ``NEO4J_PASSWORD`` (and set ``NEO4J_AUTH`` on
    # the compose service to match).
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "devpassword"
    # Default k-hop expansion depth for query-side traversal (B3 is finalized in
    # V6; a sensible default lives here so the GraphStore has one). 2 hops trades
    # multi-hop recall against subgraph noise.
    khop_depth: int = 2

    # --- Query-side retrieval seeding + ranking (V6, ADR-0007) --------------
    # B5: how many kNN seeds to pull per index before k-hop expansion. Entity
    # seeds anchor the graph traversal; sentence seeds anchor the supporting
    # evidence. Small for a demo corpus; tune for recall vs. subgraph noise.
    seed_top_k_entities: int = 5
    seed_top_k_sentences: int = 5
    # B4: the subgraph ranking-function weights (graph_rag.query.ranking).
    # ``score = rank_weight_seed * seed_similarity + rank_weight_proximity *
    # 1/(1+hop_distance)``. Fixed for benchmark reproducibility (ADR-0009); tune
    # via ``RANK_WEIGHT_SEED`` / ``RANK_WEIGHT_PROXIMITY``.
    rank_weight_seed: float = 0.7
    rank_weight_proximity: float = 0.3

    # --- Logging seam --------------------------------------------------------
    log_level: str = "INFO"
    # ``json`` writes one object per line (time, level, logger, message, trace id) for a log
    # collector; ``text`` is the default. Content stays out of both (docs/LOG-POLICY.md).
    log_format: str = "text"

    # --- Metrics and traces (GRA-56) ----------------------------------------
    # ``/metrics`` (Prometheus text) is served only when on. With ``require_auth`` it needs
    # ``metrics_token`` as a bearer token, never a tenant's API key. The worker serves the same on
    # its health port. Spans are exported over OTLP/HTTP only when an endpoint is set (needs the
    # ``otel`` extra); unset, tracing is a no-op.
    metrics_enabled: bool = False
    metrics_token: str = ""
    otel_exporter_otlp_endpoint: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (constructed once per process)."""
    return Settings()
