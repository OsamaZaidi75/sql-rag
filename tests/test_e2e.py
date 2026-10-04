"""End-to-End integration tests with Ollama and SQLite"""

import pytest
from core.database import init_sample_database, execute_safe_query, DEFAULT_DB_PATH
from core.vector_store import VectorStore, index_database_schema
from core.rag_engine import generate_sql_query, summarize_results_nl


def _ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server not reachable; skipping live-LLM e2e tests",
)


@pytest.fixture(scope="module")
def setup_env():
    init_sample_database(DEFAULT_DB_PATH, force_recreate=True)
    vstore = VectorStore(backend="auto")
    index_database_schema(DEFAULT_DB_PATH, vstore)
    return vstore


def test_e2e_top_salaries(setup_env):
    vstore = setup_env
    query = "List the top 3 highest paid employees with their name and salary"
    sql, chunks, raw_output = generate_sql_query(
        nl_query=query,
        db_path=DEFAULT_DB_PATH,
        vector_store=vstore,
        model_name="llama3.1:latest",
        top_k=4
    )
    assert "SELECT" in sql.upper()
    df, msg, _ = execute_safe_query(DEFAULT_DB_PATH, sql)
    assert not df.empty
    assert len(df) == 3


def test_e2e_join_aggregation(setup_env):
    vstore = setup_env
    query = "What is the total sales revenue per region?"
    sql, chunks, raw_output = generate_sql_query(
        nl_query=query,
        db_path=DEFAULT_DB_PATH,
        vector_store=vstore,
        model_name="llama3.1:latest",
        top_k=5
    )
    assert "SELECT" in sql.upper()
    df, msg, _ = execute_safe_query(DEFAULT_DB_PATH, sql)
    assert "✅" in msg
    assert not df.empty
    assert "region" in [c.lower() for c in df.columns]
