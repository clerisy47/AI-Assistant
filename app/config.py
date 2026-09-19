"""Central configuration, loaded from environment variables / a .env file.

Every tunable mentioned in the project brief (provider choice, temperature,
top_p, chunk size, retrieval depth, ...) lives here so it's changed in one
place, without touching code.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider selection -------------------------------------------------
    # "anthropic" | "openai" | "local"  (local = vLLM, via the OpenAI-compatible provider)
    LLM_PROVIDER: str = "anthropic"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-5"

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    VLLM_BASE_URL: str = "http://vllm:8000/v1"
    VLLM_MODEL: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # --- Prompt engineering / generation defaults --------------------------------
    # Sensible defaults for open-ended chat; individual endpoints override these
    # where determinism matters more than variety (see app/api/rag.py, structured.py).
    TEMPERATURE: float = 0.7
    TOP_P: float = 1.0
    MAX_TOKENS: int = 1500
    MAX_TOOL_ITERATIONS: int = 5

    # --- RAG ----------------------------------------------------------------------
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "knowledge_base"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 4

    # --- Verified research / context engineering ------------------------------------
    # Caps keep multi-search dumps from saturating the research agent context;
    # SKILLS_DIR is the progressive-disclosure root (used from Phase 2).
    EVIDENCE_EXCERPT_MAX_CHARS: int = 500
    TOOL_RESULT_MAX_CHARS: int = 2000
    SKILLS_DIR: str = "skills"
    MAX_RESEARCH_ITERATIONS: int = 8  # supervisor research↔verify rounds
    MAX_RESEARCH_TOOL_CALLS: int = 12  # global tool-call budget across research passes
    MAX_RESEARCH_PASS_ITERATIONS: int = 4  # LLM turns per research pass inside supervisor
    # Failure injection for eval/demo: "" | kb_unavailable | kb_timeout | kb_malformed
    INJECT_FAILURE: str = ""

    # --- App ------------------------------------------------------------------------
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8080
    LOG_LEVEL: str = "INFO"


settings = Settings()
