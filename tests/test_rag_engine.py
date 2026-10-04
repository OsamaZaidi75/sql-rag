"""Unit tests for core.rag_engine"""

import pytest
from core.rag_engine import extract_sql_from_response


def test_extract_sql_from_markdown_block():
    raw_1 = """Here is the SQL query you requested:
```sql
SELECT name, salary FROM employees WHERE salary > 100000;
```
Let me know if you need anything else!"""
    sql = extract_sql_from_response(raw_1)
    assert sql == "SELECT name, salary FROM employees WHERE salary > 100000;"


def test_extract_sql_with_think_tags():
    raw_2 = """<think>
We need to get top 5 salaries from employees table ordered by salary descending.
</think>
```sqlite
SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5;
```"""
    sql = extract_sql_from_response(raw_2)
    assert sql == "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5;"


def test_extract_sql_raw_lines():
    raw_3 = """SELECT d.name, COUNT(e.id) AS emp_count
FROM departments d
JOIN employees e ON d.id = e.department_id
GROUP BY d.name;"""
    sql = extract_sql_from_response(raw_3)
    assert "SELECT d.name, COUNT(e.id)" in sql
    assert "JOIN employees" in sql
