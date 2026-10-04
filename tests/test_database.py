"""Unit tests for core.database"""

import os
import sqlite3
import pandas as pd
import pytest
from core.database import (
    init_sample_database,
    get_database_tables,
    get_table_details,
    get_full_database_schema,
    validate_sql_safety,
    execute_safe_query,
    load_csv_to_sqlite
)

TEST_DB = "data/test_sample.db"


@pytest.fixture(autouse=True)
def setup_teardown():
    init_sample_database(TEST_DB, force_recreate=True)
    yield
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


def test_init_sample_database():
    assert os.path.exists(TEST_DB)
    tables = get_database_tables(TEST_DB)
    assert "departments" in tables
    assert "employees" in tables
    assert "projects" in tables
    assert "sales_transactions" in tables


def test_get_table_details():
    details = get_table_details(TEST_DB, "employees")
    assert details["table_name"] == "employees"
    assert details["row_count"] > 0
    col_names = [c["name"] for c in details["columns"]]
    assert "salary" in col_names
    assert "department_id" in col_names
    assert len(details["foreign_keys"]) >= 1


def test_validate_sql_safety():
    # Valid queries
    assert validate_sql_safety("SELECT * FROM employees")[0] is True
    assert validate_sql_safety("WITH t AS (SELECT * FROM employees) SELECT * FROM t")[0] is True
    assert validate_sql_safety("EXPLAIN QUERY PLAN SELECT * FROM employees")[0] is True

    # Blocked dangerous DDL/DML queries
    assert validate_sql_safety("DROP TABLE employees")[0] is False
    assert validate_sql_safety("DELETE FROM employees WHERE id = 1")[0] is False
    assert validate_sql_safety("UPDATE employees SET salary = 999999")[0] is False
    assert validate_sql_safety("INSERT INTO employees VALUES (99, 'Hacker', 1, 'x', 1, '2020-01-01', 'x', 'x', 5)")[0] is False
    assert validate_sql_safety("ALTER TABLE employees ADD COLUMN is_admin INTEGER")[0] is False
    assert validate_sql_safety("ATTACH DATABASE 'malicious.db' AS evil")[0] is False


def test_execute_safe_query():
    # Valid execution
    df, status, latency = execute_safe_query(TEST_DB, "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3")
    assert len(df) == 3
    assert "name" in df.columns
    assert "salary" in df.columns
    assert "✅" in status
    assert latency > 0

    # Blocked query execution
    df_blocked, status_blocked, _ = execute_safe_query(TEST_DB, "DROP TABLE departments")
    assert df_blocked.empty
    assert "⚠️" in status_blocked


def test_load_csv_to_sqlite(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("item,price,quantity\nWidget,19.99,10\nGadget,29.99,5\n")

    custom_db = str(tmp_path / "custom.db")
    load_csv_to_sqlite(str(csv_file), table_name="inventory", db_path=custom_db)

    tables = get_database_tables(custom_db)
    assert "inventory" in tables

    df, _, _ = execute_safe_query(custom_db, "SELECT * FROM inventory")
    assert len(df) == 2
    assert "price" in df.columns
