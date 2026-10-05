"""Tests for the hardened safety layer: keyword validation, SQLite authorizer,
LIMIT injection, and structural (sqlglot) validation."""

import os
import sqlite3

import pytest

from core.database import (
    init_sample_database,
    get_database_tables,
    get_full_database_schema,
    validate_sql_safety,
    validate_sql_structure,
    ensure_limit,
    execute_safe_query,
    _read_only_authorizer,
)

TEST_DB = "data/test_safety.db"


@pytest.fixture(autouse=True)
def setup_teardown():
    init_sample_database(TEST_DB, force_recreate=True)
    yield
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


def test_keyword_validator_blocks_writes():
    for bad in [
        "DROP TABLE employees",
        "DELETE FROM employees",
        "UPDATE employees SET salary = 0",
        "INSERT INTO employees VALUES (1)",
        "ALTER TABLE employees ADD COLUMN x TEXT",
        "SELECT * FROM employees; DROP TABLE departments",
        "/* sneaky */ DELETE FROM sales_transactions",
    ]:
        ok, _msg = validate_sql_safety(bad)
        assert not ok, f"should have blocked: {bad}"


def test_keyword_validator_allows_select():
    ok, _ = validate_sql_safety("SELECT name, salary FROM employees WHERE salary > 100000")
    assert ok
    ok, _ = validate_sql_safety("WITH cte AS (SELECT id FROM employees) SELECT * FROM cte")
    assert ok


def test_authorizer_denies_writes_allows_reads():
    assert _read_only_authorizer(sqlite3.SQLITE_SELECT, None, None, None, None) == sqlite3.SQLITE_OK
    assert _read_only_authorizer(sqlite3.SQLITE_READ, None, None, None, None) == sqlite3.SQLITE_OK
    for action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
                   sqlite3.SQLITE_DELETE, sqlite3.SQLITE_CREATE_TABLE,
                   sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_ALTER_TABLE):
        assert _read_only_authorizer(action, None, None, None, None) == sqlite3.SQLITE_DENY


def test_execute_blocks_destructive_query():
    df, msg, _ = execute_safe_query(TEST_DB, "DROP TABLE employees")
    assert df.empty
    assert "⚠️" in msg
    # table must still exist
    assert "employees" in get_database_tables(TEST_DB)


def test_ensure_limit_injects_missing_limit():
    out = ensure_limit("SELECT * FROM employees", 500)
    assert "LIMIT 500" in out.upper()


def test_ensure_limit_respects_existing_limit():
    out = ensure_limit("SELECT * FROM employees LIMIT 5", 500)
    assert "LIMIT 5" in out.upper()
    assert "500" not in out


def test_execute_enforces_row_cap():
    df, msg, _ = execute_safe_query(TEST_DB, "SELECT * FROM employees", max_rows=3)
    assert len(df) <= 3
    assert "✅" in msg


def test_structural_validation_accepts_good_sql():
    schema = get_full_database_schema(TEST_DB)
    ok, errors = validate_sql_structure(
        "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id "
        "WHERE e.salary > 100000 ORDER BY e.salary DESC LIMIT 5",
        schema,
    )
    assert ok, errors


def test_structural_validation_catches_hallucinated_column():
    schema = get_full_database_schema(TEST_DB)
    ok, errors = validate_sql_structure("SELECT bonus FROM employees", schema)
    assert not ok
    assert any("bonus" in e for e in errors)


def test_structural_validation_catches_hallucinated_table():
    schema = get_full_database_schema(TEST_DB)
    ok, errors = validate_sql_structure("SELECT * FROM emp", schema)
    assert not ok
    assert any("emp" in e for e in errors)


def test_structural_validation_handles_cte_and_aliases():
    schema = get_full_database_schema(TEST_DB)
    ok, errors = validate_sql_structure(
        "WITH rich AS (SELECT name, salary FROM employees WHERE salary > 120000) "
        "SELECT name FROM rich ORDER BY name",
        schema,
    )
    assert ok, errors
    ok, errors = validate_sql_structure(
        "SELECT region, SUM(amount) AS total FROM sales_transactions "
        "GROUP BY region ORDER BY total DESC",
        schema,
    )
    assert ok, errors


def test_structural_validation_rejects_non_select():
    schema = get_full_database_schema(TEST_DB)
    ok, errors = validate_sql_structure("DELETE FROM employees", schema)
    assert not ok
