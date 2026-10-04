"""
Database Manager for SQL RAG
Handles SQLite connection, schema inspection, sample database generation,
CSV ingestion, and safe read-only SQL execution.
"""

import os
import re
import time
import sqlite3
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path


DEFAULT_DB_PATH = "data/sample.db"

# Forbidden SQL keywords / patterns for security
FORBIDDEN_KEYWORDS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bUPDATE\b",
    r"\bINSERT\b",
    r"\bALTER\b",
    r"\bCREATE\b",
    r"\bTRUNCATE\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bREINDEX\b",
    r"\bVACUUM\b",
    r"\bPRAGMA\b(?!\s*table_info|\s*foreign_key_list)",
]


def init_sample_database(db_path: str = DEFAULT_DB_PATH, force_recreate: bool = False) -> str:
    """Initializes a rich sample relational database with 4 interconnected tables."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    if os.path.exists(db_path) and not force_recreate:
        return db_path

    if os.path.exists(db_path) and force_recreate:
        try:
            os.remove(db_path)
        except Exception:
            pass

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Departments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS departments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        budget REAL NOT NULL,
        location TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (date('now'))
    );
    """)

    departments = [
        (1, "Engineering", 1200000.0, "San Francisco", "2020-01-15"),
        (2, "Marketing", 450000.0, "New York", "2020-03-01"),
        (3, "Sales", 850000.0, "Chicago", "2020-02-10"),
        (4, "Human Resources", 250000.0, "San Francisco", "2020-04-20"),
        (5, "Product & Design", 600000.0, "Austin", "2021-06-01"),
        (6, "Finance", 350000.0, "New York", "2020-01-01"),
    ]
    cursor.executemany("INSERT OR REPLACE INTO departments VALUES (?,?,?,?,?)", departments)

    # 2. Employees table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        salary REAL NOT NULL,
        hire_date TEXT NOT NULL,
        city TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        performance_rating REAL NOT NULL,
        FOREIGN KEY (department_id) REFERENCES departments (id)
    );
    """)

    employees = [
        (1, "Alice Johnson", 1, "Senior AI Engineer", 145000.0, "2021-03-15", "San Francisco", "alice@enterprise.com", 4.8),
        (2, "Bob Smith", 2, "Marketing Director", 115000.0, "2020-07-20", "New York", "bob@enterprise.com", 4.2),
        (3, "Carol White", 1, "Lead Systems Architect", 165000.0, "2019-11-01", "San Francisco", "carol@enterprise.com", 4.9),
        (4, "David Brown", 3, "Account Executive", 88000.0, "2022-01-10", "Chicago", "david@enterprise.com", 3.9),
        (5, "Eve Davis", 4, "HR Business Partner", 82000.0, "2021-09-05", "San Francisco", "eve@enterprise.com", 4.5),
        (6, "Frank Miller", 1, "Full Stack Developer", 110000.0, "2023-02-28", "Austin", "frank@enterprise.com", 4.1),
        (7, "Grace Lee", 3, "Senior Sales Manager", 125000.0, "2020-12-12", "Chicago", "grace@enterprise.com", 4.7),
        (8, "Hank Wilson", 2, "Growth Analyst", 78000.0, "2022-06-18", "New York", "hank@enterprise.com", 4.0),
        (9, "Ivy Chen", 5, "Principal Product Manager", 150000.0, "2021-08-14", "Austin", "ivy@enterprise.com", 4.6),
        (10, "Jack Taylor", 6, "Senior Financial Analyst", 102000.0, "2020-05-19", "New York", "jack@enterprise.com", 4.3),
        (11, "Karen Scott", 1, "DevOps Specialist", 128000.0, "2022-10-01", "Seattle", "karen@enterprise.com", 4.4),
        (12, "Leo Martinez", 5, "UI/UX Designer", 98000.0, "2023-04-15", "Austin", "leo@enterprise.com", 4.0),
    ]
    cursor.executemany("INSERT OR REPLACE INTO employees VALUES (?,?,?,?,?,?,?,?,?)", employees)

    # 3. Projects table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department_id INTEGER NOT NULL,
        lead_id INTEGER NOT NULL,
        budget REAL NOT NULL,
        status TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT,
        FOREIGN KEY (department_id) REFERENCES departments (id),
        FOREIGN KEY (lead_id) REFERENCES employees (id)
    );
    """)

    projects = [
        (1, "Project Titan - Cloud Migration", 1, 3, 350000.0, "Completed", "2023-01-10", "2023-11-30"),
        (2, "AI Knowledge Assistant", 1, 1, 280000.0, "In Progress", "2023-06-01", "2024-06-30"),
        (3, "Q4 Brand Redesign Campaign", 2, 2, 120000.0, "Completed", "2023-09-01", "2023-12-31"),
        (4, "Enterprise CRM Rollout", 3, 7, 200000.0, "In Progress", "2023-08-15", "2024-04-30"),
        (5, "Next-Gen Mobile Portal", 5, 9, 310000.0, "Planning", "2024-01-15", "2024-12-31"),
        (6, "Annual Financial Audit System", 6, 10, 95000.0, "Completed", "2023-03-01", "2023-09-15"),
        (7, "Global Employee Onboarding v2", 4, 5, 60000.0, "In Progress", "2023-10-01", "2024-03-31"),
    ]
    cursor.executemany("INSERT OR REPLACE INTO projects VALUES (?,?,?,?,?,?,?,?)", projects)

    # 4. Sales Transactions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        customer_name TEXT NOT NULL,
        product_category TEXT NOT NULL,
        amount REAL NOT NULL,
        units_sold INTEGER NOT NULL,
        transaction_date TEXT NOT NULL,
        region TEXT NOT NULL,
        FOREIGN KEY (employee_id) REFERENCES employees (id)
    );
    """)

    sales = [
        (1, 4, "Acme Corporation", "Enterprise SaaS", 45000.0, 1, "2023-01-15", "North America"),
        (2, 7, "Globex International", "Cloud Storage", 72000.0, 3, "2023-02-20", "North America"),
        (3, 4, "Initech Systems", "Security Suite", 28000.0, 2, "2023-03-11", "Europe"),
        (4, 7, "Umbrella Biotech", "Enterprise SaaS", 110000.0, 5, "2023-04-05", "North America"),
        (5, 4, "Stark Industries", "AI Analytics", 95000.0, 2, "2023-05-18", "Asia-Pacific"),
        (6, 7, "Wayne Enterprises", "Cloud Storage", 64000.0, 4, "2023-06-22", "North America"),
        (7, 4, "Cyberdyne Corp", "Security Suite", 52000.0, 2, "2023-07-30", "Europe"),
        (8, 7, "Massive Dynamic", "Enterprise SaaS", 88000.0, 4, "2023-08-14", "North America"),
        (9, 4, "Hooli Tech", "AI Analytics", 130000.0, 3, "2023-09-09", "North America"),
        (10, 7, "Soylent Corp", "Cloud Storage", 41000.0, 2, "2023-10-25", "Latin America"),
        (11, 4, "Aperture Science", "Enterprise SaaS", 79000.0, 3, "2023-11-12", "Europe"),
        (12, 7, "Pied Piper", "AI Analytics", 105000.0, 4, "2023-12-05", "North America"),
    ]
    cursor.executemany("INSERT OR REPLACE INTO sales_transactions VALUES (?,?,?,?,?,?,?,?)", sales)

    conn.commit()
    conn.close()
    return db_path


def load_csv_to_sqlite(csv_file, table_name: str = "uploaded_data", db_path: str = "data/custom.db") -> str:
    """Imports an uploaded CSV file into a SQLite database."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    df = pd.read_csv(csv_file)

    # Clean column names for SQL safety
    df.columns = [re.sub(r"[^\w]", "_", str(col)).strip("_") for col in df.columns]

    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return db_path


def get_database_tables(db_path: str) -> List[str]:
    """Returns list of user tables in the SQLite database."""
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []


def get_table_details(db_path: str, table_name: str) -> Dict[str, Any]:
    """Retrieves full metadata for a table including columns, types, foreign keys, row count, and distinct values."""
    if not os.path.exists(db_path):
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Column info
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    cols = cursor.fetchall()
    columns = [
        {
            "cid": c[0],
            "name": c[1],
            "type": c[2] or "TEXT",
            "notnull": bool(c[3]),
            "default_value": c[4],
            "is_pk": bool(c[5]),
        }
        for c in cols
    ]

    # Foreign keys
    cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
    fks_raw = cursor.fetchall()
    foreign_keys = [
        {
            "id": fk[0],
            "from_column": fk[3],
            "to_table": fk[2],
            "to_column": fk[4],
        }
        for fk in fks_raw
    ]

    # Row count
    try:
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = cursor.fetchone()[0]
    except Exception:
        row_count = 0

    # Sample distinct categorical values for string columns
    sample_values = {}
    for col in columns:
        col_name = col["name"]
        try:
            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL LIMIT 6')
            vals = [row[0] for row in cursor.fetchall()]
            if vals:
                sample_values[col_name] = vals
        except Exception:
            pass

    # Sample rows
    try:
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
        col_names = [c["name"] for c in columns]
        sample_rows = [dict(zip(col_names, row)) for row in cursor.fetchall()]
    except Exception:
        sample_rows = []

    conn.close()
    return {
        "table_name": table_name,
        "columns": columns,
        "foreign_keys": foreign_keys,
        "row_count": row_count,
        "sample_values": sample_values,
        "sample_rows": sample_rows,
    }


def get_full_database_schema(db_path: str) -> Dict[str, Any]:
    """Extracts a comprehensive schema map of the entire database."""
    tables = get_database_tables(db_path)
    schema_map = {}
    for table in tables:
        schema_map[table] = get_table_details(db_path, table)
    return schema_map


def format_schema_for_llm(
    schema_map: Dict[str, Any],
    include_samples: bool = True,
    only_tables: Optional[List[str]] = None,
) -> str:
    """Formats the schema into a concise, high-signal representation for LLM context.

    When ``only_tables`` is given, only those tables are included — this is what
    makes retrieval load-bearing on large databases instead of dumping the whole
    schema into every prompt.
    """
    lines = []
    lines.append("=== DATABASE SCHEMA (SQLite) ===")

    tables = [t for t in schema_map if only_tables is None or t in only_tables]
    for table_name in tables:
        details = schema_map[table_name]
        col_desc = []
        for c in details["columns"]:
            pk_str = " PRIMARY KEY" if c["is_pk"] else ""
            col_desc.append(f"{c['name']} {c['type']}{pk_str}")

        lines.append(f"\nTable `{table_name}` ({details['row_count']} rows):")
        lines.append(f"  Columns: {', '.join(col_desc)}")

        if details["foreign_keys"]:
            fk_desc = [
                f"{fk['from_column']} -> {fk['to_table']}.{fk['to_column']}"
                for fk in details["foreign_keys"]
            ]
            lines.append(f"  Foreign Keys: {', '.join(fk_desc)}")

        if include_samples and details["sample_values"]:
            sample_strs = []
            for col, vals in details["sample_values"].items():
                if len(vals) > 0 and any(isinstance(v, str) for v in vals):
                    val_preview = ", ".join([repr(v) for v in vals[:4]])
                    sample_strs.append(f"{col}: [{val_preview}]")
            if sample_strs:
                lines.append(f"  Sample Values: {'; '.join(sample_strs[:5])}")

    return "\n".join(lines)


def validate_sql_safety(sql: str) -> Tuple[bool, Optional[str]]:
    """Checks if the SQL statement complies with read-only security constraints."""
    cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.MULTILINE).strip()

    if not cleaned:
        return False, "Query is empty."

    upper_sql = cleaned.upper()

    for pattern in FORBIDDEN_KEYWORDS:
        if re.search(pattern, upper_sql):
            matched = re.search(pattern, upper_sql).group(0)
            return False, f"Blocked: Operation '{matched}' is not permitted in read-only mode."

    # Check that query starts with SELECT or WITH
    first_word = cleaned.split()[0].upper()
    if first_word not in ("SELECT", "WITH", "EXPLAIN"):
        return False, f"Blocked: Queries must start with SELECT, WITH, or EXPLAIN (got '{first_word}')."

    return True, None


def _read_only_authorizer(action, arg1, arg2, db_name, source):
    """SQLite authorizer callback: allow only read operations, deny everything else.

    This is the bulletproof layer beneath the keyword validator — even if a
    crafted query slips past the regex checks, SQLite itself refuses the write.
    """
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
    }
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def ensure_limit(sql: str, max_rows: int) -> str:
    """Guarantees the query cannot return more than ``max_rows`` rows.

    Uses sqlglot to inject a LIMIT clause when the query doesn't already have
    one; falls back to a regex check + append when sqlglot is unavailable.
    """
    try:
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one(sql, read="sqlite")
        target = parsed
        if isinstance(parsed, exp.Union):
            # apply to each branch of a UNION
            changed = False
            for branch in parsed.find_all(exp.Select):
                if branch.args.get("limit") is None:
                    branch.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
                    changed = True
            return parsed.sql(dialect="sqlite") if changed else sql
        if isinstance(target, exp.Select) and target.args.get("limit") is None:
            target.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
            return target.sql(dialect="sqlite")
        return sql
    except Exception:
        pass

    # Fallback: regex check + naive append
    if re.search(r"(?i)\bLIMIT\s+\d+", sql):
        return sql
    return sql.strip().rstrip(";") + f" LIMIT {max_rows}"


def validate_sql_structure(sql: str, schema_map: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Structurally validates generated SQL against the real schema using sqlglot.

    Catches the most common LLM failure mode — hallucinated table/column names —
    *before* anything touches the database. Returns (is_valid, [error messages]).
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return True, []  # parser unavailable: skip gracefully, other layers still apply

    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:
        return False, [f"SQL parse error: {str(e)[:200]}"]

    stmt = parsed
    if not isinstance(stmt, (exp.Select, exp.Union)):
        return False, ["Only SELECT-type queries are allowed (no DDL/DML)."]

    errors: List[str] = []
    known_tables = {t.lower(): t for t in schema_map}
    col_index = {
        t.lower(): {c["name"].lower() for c in details["columns"]}
        for t, details in schema_map.items()
    }

    # CTE names and SELECT aliases are legal identifiers that aren't real tables/columns
    cte_names = {c.alias_or_name.lower() for c in parsed.find_all(exp.CTE)}
    alias_names = {a.alias_or_name.lower() for a in parsed.find_all(exp.Alias)}

    # alias-or-name -> real table (lowercased)
    scope_tables: Dict[str, str] = {}
    for tbl in parsed.find_all(exp.Table):
        real = (tbl.name or "").lower()
        if not real or real in cte_names:
            continue
        if real not in known_tables:
            errors.append(f"Unknown table `{tbl.name}` (not in database schema).")
            continue
        scope_tables[real] = real
        alias = tbl.alias_or_name.lower()
        if alias != real:
            scope_tables[alias] = real

    for col in parsed.find_all(exp.Column):
        cname = (col.name or "").lower()
        if not cname or cname == "*" or cname in alias_names:
            continue
        ctable = (col.table or "").lower()
        if ctable:
            if ctable in cte_names:
                continue
            real = scope_tables.get(ctable)
            if real is None:
                errors.append(f"Unknown table/alias `{col.table}` in `{col.sql(dialect='sqlite')}`.")
            elif cname not in col_index.get(real, set()):
                errors.append(f"Unknown column `{col.name}` in table `{real}`.")
        else:
            candidates = [
                t for t in set(scope_tables.values())
                if cname in col_index.get(t, set())
            ]
            if not candidates:
                errors.append(
                    f"Unknown column `{col.name}` (not found in any referenced table)."
                )

    return (len(errors) == 0), errors


def explain_query_plan(db_path: str, sql: str) -> str:
    """Returns SQLite's EXPLAIN QUERY PLAN output for a query (read-only)."""
    ok, err = validate_sql_safety(sql)
    if not ok:
        return f"Blocked: {err}"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("EXPLAIN QUERY PLAN " + sql.strip().rstrip(";"))
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return "(no plan rows returned)"
        return "\n".join(f"{r[0]}|{r[1]}|{r[2]}|{r[3]}" for r in rows)
    except Exception as e:
        return f"EXPLAIN failed: {e}"


def execute_safe_query(
    db_path: str,
    sql: str,
    max_rows: int = 500,
    timeout_secs: float = 15.0,
) -> Tuple[pd.DataFrame, str, float]:
    """
    Executes a read-only SQL query against SQLite with layered safety:
    keyword validation -> LIMIT injection -> SQLite authorizer (deny writes
    at the engine level) -> progress-handler query timeout.
    Returns (DataFrame, status_message, latency_ms).
    """
    is_safe, error_msg = validate_sql_safety(sql)
    if not is_safe:
        return pd.DataFrame(), f"⚠️ Security Guardrail: {error_msg}", 0.0

    # Guarantee a row cap even if the LLM forgot LIMIT
    sql = ensure_limit(sql, max_rows)

    start_time = time.perf_counter()
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        # Bulletproof read-only: deny any write operation at the engine level
        conn.set_authorizer(_read_only_authorizer)
        # Kill runaway queries (e.g. accidental CROSS JOINs)
        deadline = start_time + timeout_secs

        def _progress():
            return 1 if time.perf_counter() > deadline else 0

        conn.set_progress_handler(_progress, 20000)

        df = pd.read_sql_query(sql, conn)
        conn.close()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if len(df) > max_rows:
            df = df.iloc[:max_rows]
            msg = f"✅ Query executed successfully in {elapsed_ms:.1f}ms. Returned {len(df)} rows (capped at {max_rows})."
        elif df.empty:
            msg = f"ℹ️ Query executed in {elapsed_ms:.1f}ms, but returned 0 rows."
        else:
            msg = f"✅ Query executed successfully in {elapsed_ms:.1f}ms. ({len(df)} rows returned)"

        return df, msg, elapsed_ms

    except sqlite3.OperationalError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        err_str = str(e)
        # Helpful tips for common SQLite errors
        hint = ""
        if "interrupted" in err_str.lower():
            hint = f" (query exceeded the {timeout_secs:g}s timeout — try a more selective query)"
        elif "not authorized" in err_str.lower():
            hint = " (blocked: query attempted a non-read operation)"
        elif "no such column" in err_str.lower():
            hint = " (Check table schema for correct column casing and names)"
        elif "no such table" in err_str.lower():
            hint = " (Check available tables in the sidebar schema)"
        return pd.DataFrame(), f"❌ SQLite Operational Error: {err_str}{hint}", elapsed_ms

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return pd.DataFrame(), f"❌ Execution Error: {str(e)}", elapsed_ms
