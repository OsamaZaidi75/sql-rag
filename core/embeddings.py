"""
Hosted embeddings via the NVIDIA API (OpenAI-compatible ``/embeddings``).

This replaces the old local embedding path (sentence-transformers + torch,
which alone accounted for most of the 10.5 GB Docker image). Dense vectors
now come from NVIDIA's hosted embedding models over HTTPS; the only new
requirement is the same ``NVIDIA_API_KEY`` the LLM client already uses.

Model notes (verified live 2026-11-18):

- Default ``nvidia/nemotron-3.5-embed-8b`` (4096-dim) REQUIRES an
  ``input_type``: ``"query"`` for search queries, ``"passage"`` for
  documents being indexed. Omitting it returns a 400.
- Fallback ``nvidia/nemotron-3-embed-1b`` (2048-dim) works with no
  ``input_type`` and accepts batch input.
- ``nvidia/nv-embedqa-mistral-7b-v2``, ``nvidia/embed-qa-4``,
  ``snowflake/arctic-embed-l`` and the llama-3.2 embed models all return
  404 ("Function not found for account") — do not use them.

To stay compatible with both input_type styles, a 400 response is retried
once without ``input_type`` before giving up.

The first API call can be slow (model cold start); the default timeout is
120s. All returned vectors are L2-normalized, matching the other backends,
so ``np.dot`` gives cosine similarity directly.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

import numpy as np

DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-8b"
DEFAULT_EMBEDDING_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBEDDING_TIMEOUT_SECS = 120

# Texts per API request. The schema index is ~100 chunks; batching keeps
# the number of round trips small without building oversized payloads.
EMBED_BATCH_SIZE = 32


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    if api_key:
        return api_key
    return os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("LLM_API_KEY", "")


class NvidiaEmbeddingClient:
    """Thin client for NVIDIA's OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_secs: int = EMBEDDING_TIMEOUT_SECS,
    ):
        self.model = model or os.environ.get(
            "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.api_key = _resolve_api_key(api_key)
        self.base_url = (base_url or DEFAULT_EMBEDDING_BASE_URL).rstrip("/")
        self.timeout_secs = timeout_secs
        self.embedding_dim: Optional[int] = None  # known after first call

    def _post(self, texts: List[str], input_type: Optional[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError(
                "NVIDIA embeddings need an API key (set NVIDIA_API_KEY)."
            )
        payload = {"model": self.model, "input": texts}
        if input_type is not None:
            payload["input_type"] = input_type
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_secs) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"NVIDIA embeddings HTTP {e.code}: {detail}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"NVIDIA embeddings connection failed: {e.reason}") from e
        try:
            items = sorted(body["data"], key=lambda d: d["index"])
            return [item["embedding"] for item in items]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"Unexpected embeddings response shape: {str(body)[:300]}"
            ) from e

    def embed(self, texts: List[str], input_type: str = "passage") -> np.ndarray:
        """Embed a batch of texts; returns an L2-normalized float32 array.

        ``input_type`` is ``"passage"`` for documents being indexed and
        ``"query"`` for search queries. Models that reject ``input_type``
        (e.g. the 1b fallback) get one retry without it.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        rows: List[List[float]] = []
        use_input_type: Optional[str] = input_type
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            try:
                rows.extend(self._post(batch, use_input_type))
            except RuntimeError as e:
                if use_input_type is not None and "HTTP 400" in str(e):
                    # Model does not accept input_type; retry without it
                    # and remember that for the remaining batches.
                    use_input_type = None
                    rows.extend(self._post(batch, None))
                else:
                    raise
        arr = np.array(rows, dtype=np.float32)
        if arr.shape[0] != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: got {arr.shape[0]} for {len(texts)} texts"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        arr = arr / norms
        self.embedding_dim = int(arr.shape[1])
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query (``input_type="query"``)."""
        return self.embed([query], input_type="query")
