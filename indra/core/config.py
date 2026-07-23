"""Central configuration. Pydantic Settings, environment-driven, zero hardcoded secrets.

Every tunable in INDRA lives here. Modules call :func:`get_settings` — they never read
``os.environ`` directly and never inline a threshold, weight, path, or model name.

Environment variables use the ``INDRA_`` prefix (``INDRA_LOG_LEVEL``), but the common third-party
key names also work unprefixed (``GEMINI_API_KEY``, ``GROQ_API_KEY``, ``NEO4J_PASSWORD``,
``DATABASE_URL``) so that a copied ``.env`` from another tool just works.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Environment(str, Enum):
    """Deployment environment."""

    LOCAL = "local"
    DEMO = "demo"
    PRODUCTION = "production"
    TEST = "test"


class StorageBackend(str, Enum):
    """Which storage implementations to bind.

    ``AUTO`` probes the real backends at startup and falls back to ``MEMORY`` per-store if a
    dependency is unreachable. See ``docs/DECISIONS.md`` D1 — this is what makes the demo
    un-killable.
    """

    AUTO = "auto"
    EXTERNAL = "external"
    MEMORY = "memory"


class FusionStrategy(str, Enum):
    """How vector and graph evidence are combined during GraphRAG retrieval."""

    WEIGHTED = "weighted"
    RRF = "rrf"


class Settings(BaseSettings):
    """Runtime configuration for every INDRA agent and service."""

    model_config = SettingsConfigDict(
        env_prefix="INDRA_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # ---------------------------------------------------------------- application
    app_name: str = "INDRA"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    #: Browsers treat ``localhost`` and ``127.0.0.1`` as distinct origins, so both spellings must be
    #: allowed — otherwise a dev server started on the loopback IP gets a silently dead frontend.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:4173", "http://127.0.0.1:4173",
            # The hosted console. Allowing it by default means anyone can open the public link,
            # run this API locally, and have the two find each other with no configuration.
            "https://sumeetredhu.github.io",
        ]
    )

    #: Deterministic mode: seeded stub LLM, fixed timestamps in generated data, no live network.
    #: Turn this on for the recorded demo and for CI.
    demo_mode: bool = False
    #: Force the offline path even when connectivity exists — used to rehearse field conditions.
    offline_mode: bool = False

    #: Ingest the bundled demo corpus during app startup, in-process.
    #: Needed on a memory-backed deployment where a separate seeding process would ingest into its
    #: own graph and exit. Set INDRA_BOOTSTRAP_DEMO=true on hosted instances.
    bootstrap_demo: bool = False

    # ---------------------------------------------------------------- paths
    data_dir: Path = _REPO_ROOT / "data"
    raw_dir: Path = _REPO_ROOT / "data" / "raw"
    processed_dir: Path = _REPO_ROOT / "data" / "processed"
    demo_dir: Path = _REPO_ROOT / "data" / "demo"
    cache_dir: Path = _REPO_ROOT / ".cache"
    export_dir: Path = _REPO_ROOT / "data" / "exports"

    # ---------------------------------------------------------------- storage selection
    storage_backend: StorageBackend = StorageBackend.AUTO

    neo4j_uri: str = Field(default="bolt://localhost:7687", validation_alias=AliasChoices("INDRA_NEO4J_URI", "NEO4J_URI"))
    neo4j_user: str = Field(default="neo4j", validation_alias=AliasChoices("INDRA_NEO4J_USER", "NEO4J_USER"))
    neo4j_password: SecretStr = Field(default=SecretStr("indra_dev_password"), validation_alias=AliasChoices("INDRA_NEO4J_PASSWORD", "NEO4J_PASSWORD"))
    neo4j_database: str = "neo4j"
    neo4j_timeout_s: float = 15.0

    chroma_dir: Path = _REPO_ROOT / ".cache" / "chroma"
    chroma_collection: str = "indra_chunks"
    chroma_host: str | None = None
    chroma_port: int = 8001

    database_url: str = Field(
        default=f"sqlite+aiosqlite:///{(_REPO_ROOT / '.cache' / 'indra.db').as_posix()}",
        validation_alias=AliasChoices("INDRA_DATABASE_URL", "DATABASE_URL"),
        description="Async SQLAlchemy URL. SQLite by default (D10); set to postgresql+asyncpg://… in Docker.",
    )

    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias=AliasChoices("INDRA_REDIS_URL", "REDIS_URL"))
    redis_stream_prefix: str = "indra:events"
    cache_ttl_s: int = 900

    # ---------------------------------------------------------------- LLM providers
    #: Ordered fallback chain. First provider that is configured and healthy wins.
    llm_provider_chain: list[str] = Field(default_factory=lambda: ["gemini", "groq", "ollama", "stub"])
    embedding_provider_chain: list[str] = Field(default_factory=lambda: ["gemini", "local", "hash"])

    gemini_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("INDRA_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"))
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "text-embedding-004"
    gemini_daily_budget: int = Field(default=450, description="Stay under the ~500/day free tier with headroom.")

    groq_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("INDRA_GROQ_API_KEY", "GROQ_API_KEY"))
    groq_model: str = "llama-3.3-70b-versatile"

    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices("INDRA_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"))
    anthropic_model: str = "claude-sonnet-5"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_embedding_model: str = "nomic-embed-text"

    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 32

    llm_timeout_s: float = 45.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 2048
    #: Deterministic seed used by the stub provider and by any provider that accepts one.
    llm_seed: int = 20260722

    # ---------------------------------------------------------------- ingestion
    max_upload_mb: int = 50
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
            ".xlsx", ".xls", ".csv", ".docx", ".doc", ".eml", ".msg", ".txt", ".md", ".json",
        ]
    )
    chunk_size_tokens: int = 384
    chunk_overlap_tokens: int = 50
    chunk_min_tokens: int = 40
    ocr_languages: list[str] = Field(default_factory=lambda: ["eng"])
    ocr_min_confidence: float = 0.55
    ocr_engine: Literal["tesseract", "easyocr", "auto"] = "auto"
    ingestion_concurrency: int = 4
    #: Skip re-parsing when the content hash already exists (D6).
    ingestion_idempotent: bool = True

    # ---------------------------------------------------------------- P&ID vision
    pid_detector: Literal["rule_based", "yolo", "auto"] = "rule_based"
    pid_yolo_weights: Path | None = None
    pid_min_symbol_confidence: float = 0.45
    pid_hough_threshold: int = 80
    pid_hough_min_line_length: int = 40
    pid_hough_max_line_gap: int = 8
    pid_tag_fuzzy_threshold: int = Field(default=82, ge=0, le=100, description="rapidfuzz score cutoff for tag correction (D5).")
    pid_connection_max_distance_px: int = 60

    # ---------------------------------------------------------------- retrieval / GraphRAG
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED
    retrieval_vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    retrieval_graph_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    retrieval_rrf_k: int = 60
    vector_top_k: int = 24
    graph_top_k: int = 24
    final_top_k: int = 10
    max_hops: int = Field(default=3, ge=1, le=4)
    context_window_tokens: int = 6000
    min_relevance_score: float = 0.15
    #: Graph-boost sub-weights: entity overlap, centrality, relationship confidence, recency.
    graph_boost_weights: dict[str, float] = Field(
        default_factory=lambda: {"entity_overlap": 0.4, "centrality": 0.2, "confidence": 0.2, "recency": 0.2}
    )
    recency_half_life_days: float = 180.0

    # ---------------------------------------------------------------- proactive intelligence
    proactive_scan_interval_s: int = 900
    maintenance_lookback_days: int = 90
    inspection_lookback_days: int = 180
    shift_log_lookback_days: int = 30
    precursor_similarity_threshold: float = 0.72
    oem_threshold_warning_ratio: float = Field(default=0.85, description="Fraction of the OEM limit that triggers a WARNING.")
    fleet_failure_min_count: int = 2
    knowledge_cliff_critical_score: float = 75.0
    retirement_horizon_days: int = 720
    alert_dedupe_window_s: int = 3600

    # ---------------------------------------------------------------- mobile
    supported_languages: list[str] = Field(default_factory=lambda: ["en", "hi", "ta", "kn", "mr"])
    default_language: str = "en"
    whisper_model: str = "base"
    tts_engine: Literal["gtts", "pyttsx3", "none"] = "gtts"
    offline_bundle_mb: int = 500
    offline_priority_order: list[str] = Field(default_factory=lambda: ["A", "B", "C"])
    photo_max_dimension_px: int = 1600

    # ---------------------------------------------------------------- compliance
    compliance_scan_cron: str = "0 2 * * *"
    compliance_deadline_warning_days: int = 30
    regulations: list[str] = Field(
        default_factory=lambda: [
            "Factory Act 1948",
            "OISD-STD-118",
            "OISD-STD-144",
            "DGMS Circulars",
            "PESO Rules",
            "Environmental Clearance",
        ]
    )

    # ---------------------------------------------------------------- security
    api_keys: list[SecretStr] = Field(default_factory=list, description="Empty disables auth (local dev only).")
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 30
    request_timeout_s: float = 120.0

    # ---------------------------------------------------------------- observability
    metrics_enabled: bool = True
    trace_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    # ---------------------------------------------------------------- validators
    @field_validator("cors_origins", "allowed_extensions", "supported_languages",
                     "llm_provider_chain", "embedding_provider_chain", "regulations",
                     "offline_priority_order", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept ``a,b,c`` as well as a JSON array from the environment."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("retrieval_graph_weight")
    @classmethod
    def _weights_sum_to_one(cls, value: float, info) -> float:  # type: ignore[no-untyped-def]
        vector = info.data.get("retrieval_vector_weight", 0.6)
        total = vector + value
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"retrieval_vector_weight + retrieval_graph_weight must equal 1.0, got {total:.3f}"
            )
        return value

    @field_validator("graph_boost_weights")
    @classmethod
    def _boost_weights_valid(cls, value: dict[str, float]) -> dict[str, float]:
        required = {"entity_overlap", "centrality", "confidence", "recency"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"graph_boost_weights missing keys: {sorted(missing)}")
        total = sum(value.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"graph_boost_weights must sum to 1.0, got {total:.3f}")
        return value

    # ---------------------------------------------------------------- derived
    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_test(self) -> bool:
        """True when running under pytest or an explicitly test environment."""
        return self.environment is Environment.TEST or "PYTEST_CURRENT_TEST" in os.environ

    @computed_field  # type: ignore[prop-decorator]
    @property
    def deterministic(self) -> bool:
        """True when output must be reproducible: demo recording or test run."""
        return self.demo_mode or self.is_test

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @computed_field  # type: ignore[prop-decorator]
    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    def secret(self, name: str) -> str | None:
        """Return a secret's plaintext value, or ``None`` if unset.

        Centralising this keeps ``.get_secret_value()`` out of business logic and makes it easy to
        audit every place a credential is read.
        """
        value = getattr(self, name, None)
        if isinstance(value, SecretStr):
            revealed = value.get_secret_value()
            return revealed or None
        return value if isinstance(value, str) and value else None

    def ensure_directories(self) -> None:
        """Create every directory INDRA writes to. Called once during startup."""
        for path in (
            self.data_dir, self.raw_dir, self.processed_dir, self.demo_dir,
            self.cache_dir, self.export_dir, self.chroma_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached, so import it freely. Tests override with
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()


def reload_settings() -> Settings:
    """Drop the cache and re-read the environment. Test helper."""
    get_settings.cache_clear()
    return get_settings()


__all__ = [
    "Environment",
    "FusionStrategy",
    "Settings",
    "StorageBackend",
    "get_settings",
    "reload_settings",
]
