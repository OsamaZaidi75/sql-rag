"""API tests for the FastAPI backend (api/).

All LLM calls go through a stub — no provider, no key, no network. The stub
returns a fixed valid query for generation and a fixed sentence for
summaries, so these tests exercise the real pipeline (retrieve -> generate
-> validate -> execute -> summarize) end to end.
"""

import time

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.config import Settings
from api.engine import BoundedLLM, Engine
from core.llm import LLMClient, LLMConfig

TOP3_SQL = "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3"


class StubLLM:
    """Deterministic stand-in: valid SQL for generation, a sentence otherwise."""

    label = "stub:test"

    def chat(self, messages, temperature=0.0, max_tokens=None):
        prompt = messages[-1]["content"]
        if "Generate the SQL query" in prompt:
            return f"```sql\n{TOP3_SQL}\n```"
        if "Provide a concise" in prompt:
            return "Carol White is the highest paid employee."
        return "```sql\nSELECT 1\n```"


class RecordingLLM(LLMClient):
    """Real client shape that records the max_tokens it was called with."""

    def __init__(self, provider):
        super().__init__(LLMConfig(provider=provider, model="test-model"))
        self.seen_max_tokens = None

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.seen_max_tokens = max_tokens
        return "ok"


@pytest.fixture()
def test_engine(tmp_path, monkeypatch):
    """Engine on a throwaway DB with the stub LLM; also swapped into the app."""
    monkeypatch.setenv("SQLRAG_DB_PATH", str(tmp_path / "sample.db"))
    settings = Settings(
        db_path=str(tmp_path / "sample.db"),
        llm_provider="cloud",
        llm_model="stub-model",
        embedding_backend="tfidf",
    )
    eng = Engine(settings=settings)
    eng.llm_override = StubLLM()
    monkeypatch.setattr(main_module, "engine", eng)
    return eng


@pytest.fixture()
def client(test_engine):
    return TestClient(main_module.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["table_count"] == 4
    assert set(body["tables"]) == {
        "departments", "employees", "projects", "sales_transactions",
    }
    assert body["llm_available"] is True  # stub override counts as available


def test_schema(client):
    r = client.get("/api/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["database"] == "sample.db"
    employees = next(t for t in body["tables"] if t["name"] == "employees")
    assert employees["row_count"] == 12
    assert any(c["name"] == "salary" for c in employees["columns"])
    assert any(
        fk["to_table"] == "departments" for fk in employees["foreign_keys"]
    )


def test_schema_unknown_table_404(client):
    assert client.get("/api/schema/nope").status_code == 404


def test_ask_stubbed_end_to_end(client):
    r = client.post("/api/ask", json={"question": "Who earns the most?"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "ORDER BY salary DESC" in body["sql"]
    assert body["row_count"] == 3
    assert body["rows"][0]["name"] == "Carol White"
    assert body["rows"][0]["salary"] == 165000.0
    assert body["summary"].startswith("Carol White is the highest paid")
    assert body["tables_used"]  # retrieval ran
    assert body["gen_time_ms"] >= 0
    assert body["exec_time_ms"] >= 0


def test_ask_rejects_empty_question(client):
    assert client.post("/api/ask", json={"question": "  "}).status_code == 422
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_ask_503_when_llm_unavailable(tmp_path, monkeypatch):
    settings = Settings(
        db_path=str(tmp_path / "sample.db"),
        llm_provider="cloud",  # no key, no override -> unavailable
        llm_model="stub-model",
        llm_api_key="",
        embedding_backend="tfidf",
    )
    eng = Engine(settings=settings)
    monkeypatch.setattr(main_module, "engine", eng)
    client = TestClient(main_module.app)
    r = client.post("/api/ask", json={"question": "Who earns the most?"})
    assert r.status_code == 503
    assert "LLM unavailable" in r.json()["detail"]


def test_execute_and_blocked_write(client):
    r = client.post("/api/execute", json={"sql": "SELECT COUNT(*) AS n FROM employees"})
    assert r.status_code == 200
    assert r.json()["rows"] == [{"n": 12}]

    r = client.post("/api/execute", json={"sql": "DROP TABLE employees"})
    assert r.status_code == 200
    assert r.json()["ok"] is False  # safety guardrail, not an exception


def test_explain(client):
    r = client.post("/api/explain", json={"sql": TOP3_SQL})
    assert r.status_code == 200
    assert "SCAN" in r.json()["plan"] or "SEARCH" in r.json()["plan"]


def test_examples(client):
    r = client.get("/api/examples")
    assert r.status_code == 200
    assert len(r.json()["examples"]) == 4


def test_eval_mock_run_completes(client):
    r = client.post("/api/eval", json={"mock": True})
    assert r.status_code == 202
    run_id = r.json()["run_id"]

    deadline = time.time() + 120
    while time.time() < deadline:
        r = client.get(f"/api/eval/{run_id}")
        assert r.status_code == 200
        body = r.json()
        if body["status"] == "done":
            break
        time.sleep(1)
    assert body["status"] == "done"
    assert body["total"] == 30
    assert body["passed"] == 30
    assert body["accuracy"] == 100.0


def test_eval_unknown_run_404(client):
    assert client.get("/api/eval/doesnotexist").status_code == 404


def test_bounded_llm_enforces_generous_cap():
    """Reasoning models need room for thinking tokens before the answer."""
    inner = RecordingLLM(provider="cloud")
    bounded = BoundedLLM(inner)
    bounded.chat([{"role": "user", "content": "hi"}])
    assert inner.seen_max_tokens == 2048  # unset -> generous default
    bounded.chat([{"role": "user", "content": "hi"}], max_tokens=150)
    assert inner.seen_max_tokens == 1024  # small cap -> raised to the floor
    bounded.chat([{"role": "user", "content": "hi"}], max_tokens=4096)
    assert inner.seen_max_tokens == 4096  # large cap -> untouched


def test_bounded_llm_passthrough_for_ollama():
    inner = RecordingLLM(provider="ollama")
    bounded = BoundedLLM(inner)
    bounded.chat([{"role": "user", "content": "hi"}])
    assert inner.seen_max_tokens is None
