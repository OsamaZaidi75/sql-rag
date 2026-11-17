"""Environment-driven configuration for the SQL RAG API.

Every setting comes from an environment variable; there are no secrets in
code and no keys are ever logged or returned by the API.

LLM provider selection (``LLM_PROVIDER``):

- ``ollama`` — local Ollama server. The ``ollama`` Python client honours
  ``OLLAMA_HOST`` (default ``http://127.0.0.1:11434``). Model via
  ``LLM_MODEL`` (default ``llama3.1``).
- ``cloud`` — any OpenAI-compatible ``/chat/completions`` endpoint.
  Default is the NVIDIA API (``https://integrate.api.nvidia.com/v1``,
  model ``nvidia/nemotron-3-ultra-550b-a55b``) — the standing default for
  hosted demos. The key comes from ``NVIDIA_API_KEY`` (``LLM_API_KEY`` and
  ``AGENTROUTER_API_KEY`` are accepted as aliases). Point ``LLM_BASE_URL``
  elsewhere (e.g. AgentRouter ``https://agentrouter.org/v1``) to use a
  different OpenAI-compatible provider; ``core/llm.py`` adds AgentRouter's
  WAF-bypass headers automatically when the base URL mentions agentrouter.
- ``auto`` (default) — ``cloud`` when an API key is present in the
  environment, otherwise ``ollama``.

Embeddings (``EMBEDDING_BACKEND``, default ``tfidf``): ``tfidf`` is fully
offline and deterministic — no model downloads, which is what the deploy
target wants. ``auto`` tries Ollama embeddings (``nomic-embed-text``), then
SentenceTransformers, then TF-IDF.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from core.database import DEFAULT_DB_PATH

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
AGENTROUTER_BASE_URL = "https://agentrouter.org/v1"
DEFAULT_OLLAMA_MODEL = "llama3.1"
DEFAULT_CLOUD_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


@dataclass(frozen=True)
class Settings:
    db_path: str = DEFAULT_DB_PATH
    llm_provider: str = "ollama"  # resolved: "ollama" | "cloud"
    llm_provider_raw: str = "auto"
    llm_model: str = DEFAULT_OLLAMA_MODEL
    llm_base_url: str = NVIDIA_BASE_URL
    llm_api_key: str = ""  # never logged, never returned by the API
    llm_timeout_secs: int = 180
    embedding_backend: str = "tfidf"
    ollama_embed_model: str = "nomic-embed-text"
    max_rows: int = 500
    query_timeout_secs: float = 15.0
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def llm_label(self) -> str:
        return f"{self.llm_provider}:{self.llm_model}"

    def public_dict(self) -> dict:
        """Settings safe to expose over the API — no key material."""
        return {
            "db_path": self.db_path,
            "llm_provider": self.llm_provider,
            "llm_provider_raw": self.llm_provider_raw,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_key_configured": bool(self.llm_api_key),
            "llm_timeout_secs": self.llm_timeout_secs,
            "embedding_backend": self.embedding_backend,
            "max_rows": self.max_rows,
            "query_timeout_secs": self.query_timeout_secs,
        }


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def load_settings() -> Settings:
    """Resolve settings from the environment (see module docstring)."""
    provider_raw = os.environ.get("LLM_PROVIDER", "auto").strip().lower()
    if provider_raw not in ("auto", "ollama", "cloud"):
        provider_raw = "auto"
    api_key = (
        os.environ.get("NVIDIA_API_KEY", "")
        or os.environ.get("LLM_API_KEY", "")
        or os.environ.get("AGENTROUTER_API_KEY", "")
    )
    if provider_raw == "auto":
        provider = "cloud" if api_key else "ollama"
    else:
        provider = provider_raw

    if provider == "cloud":
        base_url = os.environ.get("LLM_BASE_URL", NVIDIA_BASE_URL)
        model = os.environ.get("LLM_MODEL", DEFAULT_CLOUD_MODEL)
    else:
        # Ollama's client reads OLLAMA_HOST itself; the base URL is unused.
        base_url = os.environ.get("LLM_BASE_URL", NVIDIA_BASE_URL)
        model = os.environ.get("LLM_MODEL", DEFAULT_OLLAMA_MODEL)

    return Settings(
        db_path=os.environ.get("SQLRAG_DB_PATH", DEFAULT_DB_PATH),
        llm_provider=provider,
        llm_provider_raw=provider_raw,
        llm_model=model,
        llm_base_url=base_url,
        llm_api_key=api_key,
        llm_timeout_secs=_env_int("LLM_TIMEOUT_SECS", 180),
        embedding_backend=os.environ.get("EMBEDDING_BACKEND", "tfidf"),
        ollama_embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        max_rows=_env_int("SQLRAG_MAX_ROWS", 500),
        query_timeout_secs=_env_float("SQLRAG_QUERY_TIMEOUT_SECS", 15.0),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8000),
    )
