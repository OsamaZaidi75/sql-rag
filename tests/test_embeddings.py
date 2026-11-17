"""Tests for the NVIDIA hosted embedding path (core/embeddings.py) and the
``nvidia`` backend of core.vector_store.VectorStore.

No real API calls: urllib is mocked for the client tests, and a keyword-based
fake client stands in for the NVIDIA API in the store tests. Live verification
against the real endpoint is a manual step (see the optimization report).
"""

import io
import json
import os
import urllib.error

import numpy as np
import pytest

from core.embeddings import NvidiaEmbeddingClient
from core.vector_store import VectorStore, get_or_build_index, index_database_schema
from core.database import init_sample_database

TEST_DB = "data/test_embeddings.db"
KEYWORDS = ["employee", "salary", "department", "budget", "sales", "revenue"]


# -- urllib mocks -------------------------------------------------------


class FakeHTTPResponse:
    """Minimal urlopen() context manager returning canned JSON."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _embed_payload(vectors):
    return {
        "data": [
            {"index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


def _fake_urlopen_factory(calls, behaviour):
    """Returns a fake urlopen recording each request and answering per behaviour.

    behaviour: list of ("ok", vectors) or ("http_error", code, body) tuples,
    consumed in order (last entry repeats).
    """

    def fake_urlopen(req, timeout=None):
        calls.append(
            {
                "url": req.full_url,
                "payload": json.loads(req.data.decode("utf-8")),
                "auth": req.get_header("Authorization"),
                "timeout": timeout,
            }
        )
        step = behaviour[min(len(calls) - 1, len(behaviour) - 1)]
        if step[0] == "http_error":
            raise urllib.error.HTTPError(
                req.full_url, step[1], "error", None,
                fp=io.BytesIO(step[2].encode("utf-8")),
            )
        return FakeHTTPResponse(_embed_payload(step[1]))

    return fake_urlopen


# -- fake embedding client for store-level tests ------------------------


class FakeNvidiaClient:
    """Deterministic keyword-count embeddings (dim = len(KEYWORDS))."""

    def __init__(self, model=None, api_key=None, base_url=None, timeout_secs=120):
        self.model = model or "fake-model"
        self.api_key = api_key or "fake-key"
        self.base_url = base_url
        self.timeout_secs = timeout_secs
        self.embedding_dim = len(KEYWORDS)
        self.calls = []

    def _vec(self, text):
        lowered = text.lower()
        vec = np.array(
            [lowered.count(kw) for kw in KEYWORDS], dtype=np.float32
        ) + 0.01
        return vec / (np.linalg.norm(vec) + 1e-9)

    def embed(self, texts, input_type="passage"):
        self.calls.append((list(texts), input_type))
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def embed_query(self, query):
        return self.embed([query], input_type="query")


class FailingNvidiaClient(FakeNvidiaClient):
    def embed(self, texts, input_type="passage"):
        raise RuntimeError("NVIDIA embeddings HTTP 503: service unavailable")

    def embed_query(self, query):
        raise RuntimeError("NVIDIA embeddings HTTP 503: service unavailable")


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setattr(
        "core.vector_store.NvidiaEmbeddingClient", FakeNvidiaClient
    )
    return FakeNvidiaClient


@pytest.fixture(autouse=True)
def _clean_db():
    yield
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


# -- client tests -------------------------------------------------------


def test_embed_posts_input_type_and_normalizes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(calls, [("ok", [[3.0, 4.0], [1.0, 0.0]])]),
    )
    client = NvidiaEmbeddingClient(model="m", api_key="test-key")
    out = client.embed(["doc one", "doc two"], input_type="passage")

    assert len(calls) == 1
    assert calls[0]["url"] == "https://integrate.api.nvidia.com/v1/embeddings"
    assert calls[0]["payload"]["model"] == "m"
    assert calls[0]["payload"]["input"] == ["doc one", "doc two"]
    assert calls[0]["payload"]["input_type"] == "passage"
    assert calls[0]["auth"] == "Bearer test-key"
    assert calls[0]["timeout"] == 120
    assert out.shape == (2, 2)
    assert client.embedding_dim == 2
    # L2-normalized: [3, 4] -> [0.6, 0.8]
    assert out[0].tolist() == pytest.approx([0.6, 0.8])
    assert out[1].tolist() == pytest.approx([1.0, 0.0])


def test_embed_query_uses_query_input_type(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(calls, [("ok", [[0.0, 2.0]])]),
    )
    client = NvidiaEmbeddingClient(model="m", api_key="test-key")
    out = client.embed_query("what is the salary?")
    assert calls[0]["payload"]["input_type"] == "query"
    assert out.shape == (1, 2)


def test_embed_retries_without_input_type_on_400(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(
            calls,
            [
                ("http_error", 400, '{"detail": "input_type not supported"}'),
                ("ok", [[1.0, 1.0]]),
            ],
        ),
    )
    client = NvidiaEmbeddingClient(model="nvidia/nemotron-3-embed-1b",
                                   api_key="test-key")
    out = client.embed(["doc"], input_type="passage")
    assert len(calls) == 2
    assert calls[0]["payload"]["input_type"] == "passage"
    assert "input_type" not in calls[1]["payload"]
    assert out.shape == (1, 2)


def test_embed_batches_large_inputs(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(
            calls,
            [
                ("ok", [[1.0, 0.0]] * 32),
                ("ok", [[1.0, 0.0]] * 3),
            ],
        ),
    )
    client = NvidiaEmbeddingClient(model="m", api_key="test-key")
    out = client.embed([f"doc {i}" for i in range(35)])
    assert len(calls) == 2
    assert len(calls[0]["payload"]["input"]) == 32
    assert len(calls[1]["payload"]["input"]) == 3
    assert out.shape == (35, 2)


def test_embed_requires_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = NvidiaEmbeddingClient(model="m", api_key="")
    with pytest.raises(RuntimeError, match="API key"):
        client.embed(["doc"])


def test_embed_empty_input_makes_no_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _fake_urlopen_factory(calls, [("ok", [])]),
    )
    client = NvidiaEmbeddingClient(model="m", api_key="test-key")
    out = client.embed([])
    assert out.shape == (0, 0)
    assert calls == []


# -- store tests --------------------------------------------------------


def test_vector_store_nvidia_backend(fake_client):
    vs = VectorStore(backend="nvidia", api_key="test-key")
    assert vs.active_backend == "nvidia"

    vs.add_chunk("Table employees with salary and department columns",
                 {"type": "schema", "table": "employees"})
    vs.add_chunk("Table sales_transactions with revenue amount",
                 {"type": "schema", "table": "sales"})
    vs.build_index()
    assert vs.embeddings.shape == (2, len(KEYWORDS))

    results = vs.search("Show me employee salaries", top_k=2)
    # The sales chunk scores below search()'s min_score filter, so only the
    # employees chunk comes back — but it must rank first.
    assert len(results) >= 1
    assert results[0][0].metadata["table"] == "employees"

    hybrid = vs.search_hybrid("employee salary", top_k=2)
    assert hybrid[0][0].metadata["table"] == "employees"


def test_vector_store_nvidia_falls_back_to_tfidf(monkeypatch):
    monkeypatch.setattr(
        "core.vector_store.NvidiaEmbeddingClient", FailingNvidiaClient
    )
    vs = VectorStore(backend="nvidia", api_key="test-key")
    assert vs.active_backend == "tfidf"  # probe failed -> offline fallback

    vs.add_chunk("Table employees with salary and department columns",
                 {"type": "schema", "table": "employees"})
    vs.build_index()
    results = vs.search("employee salaries", top_k=1)
    assert len(results) == 1


def test_vector_store_nvidia_without_key_falls_back_to_tfidf(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    vs = VectorStore(backend="nvidia", api_key="")
    assert vs.active_backend == "tfidf"


def test_unknown_embedding_backend_rejected():
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        VectorStore(backend="sentence-transformers")


def test_nvidia_cache_reused_then_rebuilt_on_model_change(fake_client, tmp_path):
    cache_dir = str(tmp_path / "cache")
    db = str(tmp_path / "c.db")
    init_sample_database(db, force_recreate=True)

    vs1 = get_or_build_index(db, backend="nvidia", cache_dir=cache_dir,
                             embedding_model="model-a", api_key="k")
    assert vs1.active_backend == "nvidia"
    manifest_path = os.path.join(cache_dir, "c.nvidia.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        assert json.load(f)["embedding_model"] == "model-a"
    mtime1 = os.path.getmtime(manifest_path)

    # Same model -> cache hit, no rebuild.
    vs2 = get_or_build_index(db, backend="nvidia", cache_dir=cache_dir,
                             embedding_model="model-a", api_key="k")
    assert os.path.getmtime(manifest_path) == mtime1
    assert len(vs2.chunks) == len(vs1.chunks)
    assert vs2.search("employee salary", top_k=1)

    # Different model -> rebuild, manifest updated.
    vs3 = get_or_build_index(db, backend="nvidia", cache_dir=cache_dir,
                             embedding_model="model-b", api_key="k")
    with open(manifest_path, encoding="utf-8") as f:
        assert json.load(f)["embedding_model"] == "model-b"
    assert len(vs3.chunks) == len(vs1.chunks)


def test_index_database_schema_with_nvidia(fake_client):
    init_sample_database(TEST_DB, force_recreate=True)
    vs = VectorStore(backend="nvidia", api_key="test-key")
    index_database_schema(TEST_DB, vs)
    assert len(vs.chunks) > 10
    results = vs.search_hybrid("department budgets", top_k=3)
    assert len(results) > 0
    assert any("department" in r[0].text.lower() for r in results)
