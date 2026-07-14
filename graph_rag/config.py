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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (constructed once per process)."""
    return Settings()
