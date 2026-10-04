"""
RAG Engine for Natural Language to SQL
Orchestrates vector retrieval, prompt compilation, LLM inference via Ollama,
SQL post-processing, automated self-correction/repair, and natural language result summarization.
"""

import re
import json
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from .vector_store import VectorStore, DocumentChunk
from .database import get_full_database_schema, format_schema_for_llm


SYSTEM_PROMPT = """You are an elite SQL engineer specializing in SQLite.
Your mission is to translate natural language questions into accurate, performant, and safe SQLite queries.

### CORE SQLITE RULES:
1. DIALECT: Generate strictly valid SQLite SQL.
2. READ-ONLY: Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or CREATE.
3. TABLE & COLUMN NAMES: Use exact table and column names from the provided schema.
4. STRING LITERALS: Use single quotes for string literals (e.g. `department = 'Engineering'`). Match sample values exactly as provided in the retrieved context.
5. JOINS: Always specify explicit ON conditions matching the foreign keys (e.g. `JOIN departments ON employees.department_id = departments.id`).
6. AGGREGATIONS & GROUP BY: Include all non-aggregated SELECT columns in the GROUP BY clause.
7. DATES: Use SQLite date functions:
   - Extract year: `strftime('%Y', date_col)`
   - Extract month: `strftime('%m', date_col)` or `strftime('%Y-%m', date_col)`
   - Current date: `date('now')`
8. CALCULATIONS: For percentage or division, avoid integer division by multiplying by 1.0 (e.g. `ROUND(100.0 * num / denom, 2)`).
9. OUTPUT: Output ONLY the SQL query enclosed in a single ```sql ... ``` block. No markdown introductory text, no reasoning, no postscript.
"""


def extract_sql_from_response(raw_text: str) -> str:
    """Robustly extracts pure SQL query from LLM response."""
    if not raw_text:
        return ""

    # Remove thinking tags from reasoning models (e.g., DeepSeek / Qwen / Llama thinking variants)
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # Match ```sql ... ```, ```sqlite ... ```, or ``` ... ``` block
    sql_match = re.search(r"```(?:[a-zA-Z0-9_-]+)?\s*\n?([\s\S]*?)\s*```", text, re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # If no code block, look for lines starting with SELECT or WITH
        lines = text.split("\n")
        sql_lines = []
        capturing = False
        for line in lines:
            stripped = line.strip()
            if not capturing and re.match(r"^(SELECT|WITH|EXPLAIN)\b", stripped, re.IGNORECASE):
                capturing = True
            if capturing:
                if stripped.startswith("---") or stripped.startswith("Explanation:"):
                    break
                sql_lines.append(line)
        if sql_lines:
            sql = "\n".join(sql_lines).strip()
        else:
            sql = text.strip()

    # Clean trailing backticks and clean semicolons
    sql = sql.strip("`").strip()
    return sql


def generate_sql_query(
    nl_query: str,
    db_path: str,
    vector_store: VectorStore,
    model_name: str = "llama3.1",
    top_k: int = 5,
    temperature: float = 0.0,
) -> Tuple[str, List[Tuple[DocumentChunk, float]], str]:
    """
    Retrieves relevant schema context and generates a SQLite query via Ollama.
    Returns: (generated_sql, retrieved_chunks, raw_llm_response).
    """
    import ollama

    # 1. RAG Retrieval from Vector Store
    retrieved_chunks = vector_store.search(nl_query, top_k=top_k)

    # 2. Extract full database schema overview
    schema_map = get_full_database_schema(db_path)
    full_schema_text = format_schema_for_llm(schema_map, include_samples=True)

    # 3. Format RAG Context Chunks
    rag_context_lines = []
    for chunk, score in retrieved_chunks:
        rag_context_lines.append(f"- [{chunk.metadata.get('type', 'info')}] (relevance: {score:.2f}): {chunk.text}")
    rag_context_text = "\n".join(rag_context_lines) if rag_context_lines else "No specific context retrieved."

    # 4. Compile User Prompt
    user_prompt = f"""=== DATABASE SCHEMA ===
{full_schema_text}

=== RELEVANT SCHEMA INSIGHTS (RAG RETRIEVAL) ===
{rag_context_text}

=== USER QUESTION ===
{nl_query}

Generate the SQL query to answer the question:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": temperature}
        )
        raw_output = response.get("message", {}).get("content", "") if isinstance(response, dict) else response.message.content
        sql = extract_sql_from_response(raw_output)
        return sql, retrieved_chunks, raw_output

    except Exception as e:
        error_msg = f"Ollama Generation Error: {str(e)}"
        return "", retrieved_chunks, error_msg


def auto_repair_sql(
    original_query: str,
    failed_sql: str,
    error_message: str,
    db_path: str,
    model_name: str = "llama3.1",
) -> Tuple[str, str]:
    """
    Self-healing mechanism: feeds failed SQL and SQLite error back to LLM to produce a corrected query.
    """
    import ollama

    schema_map = get_full_database_schema(db_path)
    schema_text = format_schema_for_llm(schema_map, include_samples=True)

    repair_prompt = f"""=== DATABASE SCHEMA ===
{schema_text}

=== ORIGINAL USER REQUEST ===
{original_query}

=== FAILED SQL QUERY ===
{failed_sql}

=== SQLITE ERROR MESSAGE ===
{error_message}

Fix the error in the SQL query so it runs successfully on SQLite. Return ONLY the corrected SQL in a ```sql ... ``` block."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": repair_prompt},
    ]

    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0.0}
        )
        raw_output = response.get("message", {}).get("content", "") if isinstance(response, dict) else response.message.content
        repaired_sql = extract_sql_from_response(raw_output)
        return repaired_sql, raw_output
    except Exception as e:
        return failed_sql, f"Repair failed: {str(e)}"


def summarize_results_nl(
    user_query: str,
    sql: str,
    df: pd.DataFrame,
    model_name: str = "llama3.1",
) -> str:
    """
    Generates a concise natural language summary of the query results.
    """
    if df.empty:
        return "The query executed successfully but found 0 matching records."

    import ollama

    # Take preview of the dataframe (up to 10 rows)
    preview_data = df.head(10).to_dict(orient="records")
    data_preview_str = json.dumps(preview_data, default=str)

    prompt = f"""User Question: {user_query}
Executed SQL: {sql}
Result Data ({len(df)} total rows, showing top {min(len(df), 10)}):
{data_preview_str}

Provide a concise, clear 1 to 3 sentence natural language summary of these results to directly answer the user's question. Focus on key numbers, names, and trends."""

    messages = [
        {"role": "system", "content": "You are a helpful business intelligence data analyst. Summarize data results concisely and accurately."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": 0.2}
        )
        summary = response.get("message", {}).get("content", "") if isinstance(response, dict) else response.message.content
        return summary.strip()
    except Exception as e:
        return f"Summary unavailable ({str(e)})"
