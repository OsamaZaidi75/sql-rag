"""Unit tests for core.vector_store"""

import pytest
from core.vector_store import VectorStore, index_database_schema
from core.database import init_sample_database

TEST_DB = "data/test_vstore.db"


@pytest.fixture(autouse=True)
def setup_teardown():
    init_sample_database(TEST_DB, force_recreate=True)
    yield
    import os
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


def test_vector_store_tfidf_backend():
    vstore = VectorStore(backend="tfidf")
    assert vstore.active_backend == "tfidf"

    vstore.add_chunk("Table employees with salary and department columns", {"type": "schema", "table": "employees"})
    vstore.add_chunk("Table sales_transactions with revenue amount", {"type": "schema", "table": "sales"})
    vstore.build_index()

    results = vstore.search("Show me employee salaries", top_k=2)
    assert len(results) > 0
    assert results[0][0].metadata["table"] == "employees"
    assert results[0][1] > 0


def test_index_database_schema():
    vstore = VectorStore(backend="tfidf")
    index_database_schema(TEST_DB, vstore)

    assert len(vstore.chunks) > 10

    # Query for departments
    results = vstore.search("What are the department budgets and locations?", top_k=3)
    assert len(results) > 0
    found_dept = any("department" in r[0].text.lower() for r in results)
    assert found_dept
