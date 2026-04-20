"""
MOSAIC — Multi-Agent Orchestration System for Analyst Intelligence and Cognition
config/settings.py

The single source of truth for every configurable value in the system.
Nothing is hardcoded anywhere else — all constants, keys, and toggles
live here and are imported wherever needed.

Why pydantic-settings and not plain os.getenv():
    pydantic-settings validates types at startup, not at runtime.
    If OPENAI_API_KEY is missing, the app crashes immediately with a
    clear error — not 3 steps into an agent run when you least want it.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Reads every value from environment variables first, then falls back
    to the defaults defined here. The .env file is loaded automatically
    by pydantic-settings — no manual dotenv loading needed elsewhere.
    """

    # ── LLM ────────────────────────────────────────────────────────────────

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    # GPT-4o is the right call here — the contradiction detector and
    # credibility scorer need strong instruction-following and long context.
    # Do not swap this for a cheaper model without re-evaluating signal quality.
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")

    # Max tokens per LLM response — 4096 is enough for structured signal output.
    # If agents start truncating, bump this before touching anything else.
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")

    # Temperature 0 — we want deterministic, factual outputs from agents.
    # This is not a creative writing system.
    openai_temperature: float = Field(default=0.0, env="OPENAI_TEMPERATURE")

    # ── LangSmith Tracing ──────────────────────────────────────────────────

    langchain_api_key: str = Field(default="", env="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="mosaic-prod", env="LANGCHAIN_PROJECT")

    # This env var is what actually switches tracing on — LangChain checks
    # for it internally. Must be the string "true", not a Python bool.
    langchain_tracing_v2: str = Field(default="true", env="LANGCHAIN_TRACING_V2")

    # ── SEC EDGAR ──────────────────────────────────────────────────────────

    # EDGAR requires this header on every request — it is not an API key,
    # just your name and email so they can contact you if you abuse the API.
    # Format must be: "First Last email@domain.com"
    sec_user_agent: str = Field(..., env="SEC_USER_AGENT")

    # EDGAR's public API base — this rarely changes but keeping it here
    # means we fix it in one place if it ever does.
    sec_base_url: str = Field(
        default="https://data.sec.gov",
        env="SEC_BASE_URL"
    )

    # EDGAR asks for a max of 10 requests per second. We stay well under
    # at one request every 0.5s — being a good citizen avoids IP blocks.
    sec_request_delay: float = Field(default=0.5, env="SEC_REQUEST_DELAY")

    sec_max_retries: int = Field(default=3, env="SEC_MAX_RETRIES")

    # ── Tickers and Filing Config ──────────────────────────────────────────

    # The six tech companies in our coverage universe.
    # CIK is SEC's internal identifier for each company — needed for all
    # EDGAR API calls. These are stable and do not change.
    ticker_cik_map: dict = Field(
        default={
            "AMZN": "0001018724",
            "MSFT": "0000789019",
            "GOOG": "0001652044",
            "ORCL": "0001341439",
            "AAPL": "0000320193",
            "NVDA": "0001045810",
        },
        env="TICKER_CIK_MAP"
    )

    # The four filing types we care about — anything else from EDGAR
    # is noise for this system.
    filing_types: list = Field(
        default=["8-K", "10-K", "10-Q", "4"],
        env="FILING_TYPES"
    )

    # 2 full fiscal years of history per ticker.
    # 8 quarters gives enough data for the temporal drift agent to
    # find meaningful signal — fewer quarters and the trends are noisy.
    lookback_years: int = Field(default=2, env="LOOKBACK_YEARS")

    # ── Data Storage ───────────────────────────────────────────────────────

    # Local paths for development — replaced with S3 URIs when we move to AWS.
    # Keeping them as plain strings means the switch requires zero code changes,
    # just an env var update.
    raw_data_path: str = Field(default="data/raw", env="RAW_DATA_PATH")
    processed_data_path: str = Field(
        default="data/processed",
        env="PROCESSED_DATA_PATH"
    )

    # ── Vector Store ───────────────────────────────────────────────────────

    # text-embedding-3-small is accurate enough for our use case and
    # significantly cheaper than text-embedding-3-large at scale.
    # 48 transcripts + 12 annual reports = a lot of chunks to embed.
    embedding_model: str = Field(
        default="text-embedding-3-small",
        env="EMBEDDING_MODEL"
    )

    # Dimension must match the model above — 1536 for text-embedding-3-small.
    # If you swap embedding models, this MUST be updated or FAISS will silently
    # produce garbage similarity scores.
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")

    # ── Chunking ───────────────────────────────────────────────────────────

    # 1000 tokens per chunk with 200 token overlap.
    # The overlap ensures a key sentence that falls at a chunk boundary
    # is not split across two chunks and lost in retrieval.
    # These values came from experimentation — do not reduce overlap below 100.
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    # ── AWS ────────────────────────────────────────────────────────────────

    # Commented out for local development — uncomment when moving to AWS.
    # aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    # s3_bucket: str = Field(..., env="S3_BUCKET")
    # rds_host: str = Field(..., env="RDS_HOST")
    # rds_port: int = Field(default=5432, env="RDS_PORT")
    # rds_db: str = Field(..., env="RDS_DB")
    # rds_user: str = Field(..., env="RDS_USER")
    # rds_password: str = Field(..., env="RDS_PASSWORD")

    # ── HITL Thresholds ────────────────────────────────────────────────────

    # Signals with confidence below this go to the human review queue.
    # 0.75 is deliberately conservative — in a financial context, a wrong
    # signal is worse than a slow one.
    hitl_confidence_threshold: float = Field(
        default=0.75,
        env="HITL_CONFIDENCE_THRESHOLD"
    )

    class Config:
        # Tells pydantic-settings to read from .env automatically.
        # The file is optional — if running in AWS, values come from
        # Secrets Manager via environment variables directly.
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allows extra fields in .env without throwing validation errors —
        # useful when the .env has vars for other tools like Docker.
        extra = "ignore"


# ── Module-level singleton ─────────────────────────────────────────────────
#
# Instantiated once when this module is first imported.
# Every other module does:
#     from config.settings import settings
# and gets this same object — no repeated env var parsing, no multiple
# instances with potentially different values.
settings = Settings()