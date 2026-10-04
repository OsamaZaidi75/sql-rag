"""Golden-set validity: every expected SQL must execute and return the
expected rows. No LLM needed — this guards the eval harness itself."""

import json
import os
from collections import Counter

import pytest

from core.database import init_sample_database, execute_safe_query, DEFAULT_DB_PATH

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "..", "evals", "golden.json")


def _norm(v):
    return round(v, 4) if isinstance(v, float) else v


def _counter(rows):
    return Counter(tuple(_norm(v) for v in row) for row in rows)


@pytest.fixture(scope="module")
def golden_db():
    init_sample_database(DEFAULT_DB_PATH, force_recreate=True)
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_golden_set_nonempty(golden_db):
    assert len(golden_db) >= 20


@pytest.mark.parametrize("idx", range(25))
def test_golden_sql_executes_and_matches(golden_db, idx):
    item = golden_db[idx]
    df, msg, _ = execute_safe_query(DEFAULT_DB_PATH, item["expected_sql"])
    assert msg.startswith("✅") or msg.startswith("ℹ️"), f"{item['id']}: {msg}"
    got = _counter(df.itertuples(index=False, name=None))
    exp = _counter(item["expected_rows"])
    assert got == exp, f"{item['id']}: result mismatch"
