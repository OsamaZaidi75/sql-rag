"""Tests for hybrid retrieval (RRF fusion), index persistence, and schema pruning."""

import json
import os

import pytest

from core.vector_store import (
    VectorStore,
    index_database_schema,
    get_or_build_index,
    rrf_fuse,
)
from core.database import init_sample_database, get_full_database_schema, DEFAULT_DB_PATH
from core.rag_engine import (
    select_relevant_tables,
    rewrite_followup,
    generate_sql_query_full,
)

TEST_DB = "data/test_retrieval.db"


@pytest.fixture(scope="module")
def vstore():
    init_sample_database(TEST_DB, force_recreate=True)
    vs = VectorStore(backend="tfidf")  # deterministic, no network
    index_database_schema(TEST_DB, vs)
    yield vs
    for p in (TEST_DB,):
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def test_rrf_fuse_orders_by_combined_rank():
    fused = rrf_fuse([[0, 1, 2], [2, 0]], k=60)
    order = [i for i, _s in fused]
    # id 0: 1/61 + 1/62 ; id 2: 1/63 + 1/61 ; id 1: 1/62  -> 0 > 2 > 1
    assert order == [0, 2, 1]
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_returns_relevant_chunks(vstore):
    results = vstore.search_hybrid("highest paid employees salary", top_k=3)
    assert len(results) == 3
    texts = " ".join(c.text for c, _ in results).lower()
    assert "employees" in texts


def test_hybrid_search_falls_back_without_bm25():
    # Self-contained store (does not mutate the shared fixture): with no BM25
    # index, hybrid search must gracefully degrade to dense-only ranking.
    vs = VectorStore(backend="tfidf")
    vs.add_chunk("Table `employees` contains salary data.", {"type": "table_summary", "table": "employees"})
    vs.add_chunk("Table `departments` contains budget data.", {"type": "table_summary", "table": "departments"})
    vs.build_index()
    vs._bm25 = None
    results = vs.search_hybrid("employee salary", top_k=2)
    assert len(results) == 2
    assert results[0][0].metadata["table"] == "employees"


def test_index_save_load_roundtrip(vstore, tmp_path):
    base = str(tmp_path / "idx")
    vstore.save_index(base, db_mtime=123.0)
    assert os.path.exists(base + ".npz")
    assert os.path.exists(base + ".chunks.json")
    assert os.path.exists(base + ".json")

    loaded = VectorStore.load_index(base)
    assert len(loaded.chunks) == len(vstore.chunks)
    assert loaded.embeddings.shape == vstore.embeddings.shape
    r1 = vstore.search_hybrid("engineering budget", top_k=3)
    r2 = loaded.search_hybrid("engineering budget", top_k=3)
    assert [c.doc_id for c, _ in r1] == [c.doc_id for c, _ in r2]


def test_get_or_build_index_caches(tmp_path):
    cache_dir = str(tmp_path / "cache")
    db = str(tmp_path / "c.db")
    init_sample_database(db, force_recreate=True)

    vs1 = get_or_build_index(db, backend="tfidf", cache_dir=cache_dir)
    manifest = os.path.join(cache_dir, "c.tfidf.json")
    assert os.path.exists(manifest)
    mtime1 = os.path.getmtime(manifest)

    vs2 = get_or_build_index(db, backend="tfidf", cache_dir=cache_dir)
    assert os.path.getmtime(manifest) == mtime1  # cache hit: no rebuild
    assert len(vs2.chunks) == len(vs1.chunks)


class FakeLLM:
    label = "fake"

    def __init__(self, sql="SELECT 1"):
        self.sql = sql

    def chat(self, messages, temperature=0.0, max_tokens=None):
        return f"```sql\n{self.sql}\n```"


def test_select_relevant_tables_picks_sales(vstore):
    schema = get_full_database_schema(TEST_DB)
    tables = select_relevant_tables(
        "What is the total sales revenue per region?", vstore, schema, top_k_tables=3
    )
    assert "sales_transactions" in tables
    assert len(tables) <= 5  # 3 + 2 FK-expansion headroom


def test_rewrite_followup_no_history_passthrough():
    llm = FakeLLM()
    q = "What is the total sales revenue?"
    assert rewrite_followup(q, None, llm) == q
    assert rewrite_followup(q, [], llm) == q


def test_generate_full_no_prune_on_small_schema(vstore):
    schema = get_full_database_schema(TEST_DB)
    assert len(schema) == 4
    full = generate_sql_query_full(
        nl_query="How many employees are there?",
        db_path=TEST_DB,
        vector_store=vstore,
        llm=FakeLLM("SELECT COUNT(*) FROM employees"),
    )
    assert full["sql"] == "SELECT COUNT(*) FROM employees"
    assert full["pruned"] is False
    assert set(full["tables_used"]) == set(schema.keys())
    assert full["rewritten_question"] == "How many employees are there?"


def test_generate_full_prunes_large_schema(vstore, tmp_path):
    # Simulate a wide schema: pruning must kick in past max_tables
    schema = get_full_database_schema(TEST_DB)
    big_schema = dict(schema)
    for i in range(10):
        big_schema[f"extra_table_{i}"] = {
            "columns": [{"name": "id", "type": "INTEGER", "is_pk": True}],
            "foreign_keys": [],
            "row_count": 0,
            "sample_values": {},
        }

    import core.rag_engine as re_mod
    orig = re_mod.get_full_database_schema
    re_mod.get_full_database_schema = lambda _p: big_schema
    try:
        full = generate_sql_query_full(
            nl_query="What is the total sales revenue per region?",
            db_path=TEST_DB,
            vector_store=vstore,
            llm=FakeLLM("SELECT 1"),
            max_tables=3,
        )
    finally:
        re_mod.get_full_database_schema = orig

    assert full["pruned"] is True
    assert "sales_transactions" in full["tables_used"]
    assert len(full["tables_used"]) <= 5
