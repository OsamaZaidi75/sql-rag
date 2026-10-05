"""
RAG Engine for Natural Language to SQL
Orchestrates vector retrieval, prompt compilation, LLM inference,
SQL post-processing, automated self-correction/repair, and natural language result summarization.

The retrieval is load-bearing: instead of dumping the whole database schema
into every prompt, hybrid retrieval selects the relevant tables first and only
those tables' DDL is injected. This is what lets the engine scale beyond toy
schemas to real databases with hundreds of tables.
"""

import re
import json
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from .vector_store import VectorStore, DocumentChunk
from .database import (
    get_full_database_schema,
    format_schema_for_llm,
    validate_sql_structure,
    execute_safe_query,
)
from .llm import LLMClient, make_llm


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

REWRITE_SYSTEM = """You rewrite a follow-up question into a standalone question, using the conversation history for context.

Rules:
- Output ONLY the rewritten question. No preamble, no quotation marks, no explanation.
- Resolve pronouns (it, they, this table) and any omitted table/column context using the history.
- If the question is already standalone and needs no history to understand, return it unchanged.
- Never answer the question. Only rewrite it."""


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


def rewrite_followup(
    question: str,
    history: Optional[List[Dict[str, str]]],
    llm: LLMClient,
) -> str:
    """
    Turns a follow-up question into a standalone question using conversation
    history, so the (stateless) retriever gets something searchable.
    Skips the LLM call when there is no history; falls back to the original
    question on any failure.
    """
    hist = [
        m for m in (history or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ]
    if not hist:
        return question

    messages = [{"role": "system", "content": REWRITE_SYSTEM}]
    messages.extend(hist[-10:])
    messages.append(
        {"role": "user", "content": f"Follow-up question: {question}\nRewritten standalone question:"}
    )
    try:
        rewritten = llm.chat(messages, temperature=0.0, max_tokens=150)
        rewritten = rewritten.strip().strip('"').strip()
        return rewritten or question
    except Exception:
        return question


def select_relevant_tables(
    question: str,
    vector_store: VectorStore,
    schema_map: Dict[str, Any],
    top_k_tables: int = 5,
) -> List[str]:
    """
    Uses hybrid retrieval to pick the tables relevant to a question, then
    expands one hop along foreign keys so joins don't lose a table.
    Returns an ordered, de-duplicated table list (capped).
    """
    try:
        results = vector_store.search_hybrid(question, top_k=top_k_tables * 4)
    except Exception:
        results = vector_store.search(question, top_k=top_k_tables * 4)

    ordered: List[str] = []
    for chunk, _score in results:
        md = chunk.metadata or {}
        for key in ("table", "from_table", "to_table"):
            t = md.get(key)
            if t and t in schema_map and t not in ordered:
                ordered.append(t)

    # One-hop FK expansion: a join needs both sides of the relationship
    expanded = list(ordered)
    for t in ordered:
        details = schema_map.get(t, {})
        for fk in details.get("foreign_keys", []):
            nt = fk.get("to_table")
            if nt and nt in schema_map and nt not in expanded:
                expanded.append(nt)
    # Reverse direction: tables that reference an already-selected table
    for t, details in schema_map.items():
        if t in expanded:
            continue
        for fk in details.get("foreign_keys", []):
            if fk.get("to_table") in ordered:
                expanded.append(t)
                break

    capped = expanded[: top_k_tables + 2]
    if not capped:
        capped = list(schema_map.keys())[:top_k_tables]
    return capped


def _resolve_llm(llm: Optional[LLMClient], model_name: str) -> LLMClient:
    return llm if llm is not None else make_llm(provider="ollama", model=model_name)


def generate_sql_query_full(
    nl_query: str,
    db_path: str,
    vector_store: VectorStore,
    model_name: str = "llama3.1",
    top_k: int = 5,
    temperature: float = 0.0,
    history: Optional[List[Dict[str, str]]] = None,
    llm: Optional[LLMClient] = None,
    prune_schema: bool = True,
    max_tables: int = 5,
) -> Dict[str, Any]:
    """
    Full generation pipeline with diagnostics. Returns a dict with:
    sql, chunks, raw_llm_response, rewritten_question, tables_used, pruned.
    """
    llm = _resolve_llm(llm, model_name)

    # 1. Resolve follow-ups into standalone questions
    rewritten_question = rewrite_followup(nl_query, history, llm)

    # 2. Hybrid RAG retrieval over the indexed schema
    try:
        retrieved_chunks = vector_store.search_hybrid(rewritten_question, top_k=top_k)
    except Exception:
        retrieved_chunks = vector_store.search(rewritten_question, top_k=top_k)

    # 3. Schema pruning: inject only the relevant tables' DDL
    schema_map = get_full_database_schema(db_path)
    pruned = prune_schema and len(schema_map) > max_tables
    if pruned:
        tables_used = select_relevant_tables(
            rewritten_question, vector_store, schema_map, top_k_tables=max_tables
        )
        full_schema_text = format_schema_for_llm(
            schema_map, include_samples=True, only_tables=tables_used
        )
        other_tables = [t for t in schema_map if t not in tables_used]
        if other_tables:
            full_schema_text += (
                f"\n\n(Other tables exist but were deemed irrelevant: "
                f"{', '.join(other_tables)}. Only use the tables above.)"
            )
    else:
        tables_used = list(schema_map.keys())
        full_schema_text = format_schema_for_llm(schema_map, include_samples=True)

    # 4. Format RAG context chunks
    rag_context_lines = []
    for chunk, score in retrieved_chunks:
        rag_context_lines.append(
            f"- [{chunk.metadata.get('type', 'info')}] (relevance: {score:.2f}): {chunk.text}"
        )
    rag_context_text = (
        "\n".join(rag_context_lines) if rag_context_lines else "No specific context retrieved."
    )

    # 5. Compile prompt and generate
    user_prompt = f"""=== DATABASE SCHEMA ===
{full_schema_text}

=== RELEVANT SCHEMA INSIGHTS (RAG RETRIEVAL) ===
{rag_context_text}

=== USER QUESTION ===
{rewritten_question}

Generate the SQL query to answer the question:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_output = llm.chat(messages, temperature=temperature)
        sql = extract_sql_from_response(raw_output)
    except Exception as e:
        raw_output = f"LLM Generation Error: {str(e)}"
        sql = ""

    return {
        "sql": sql,
        "chunks": retrieved_chunks,
        "raw_llm_response": raw_output,
        "rewritten_question": rewritten_question,
        "tables_used": tables_used,
        "pruned": pruned,
    }


def generate_sql_query(
    nl_query: str,
    db_path: str,
    vector_store: VectorStore,
    model_name: str = "llama3.1",
    top_k: int = 5,
    temperature: float = 0.0,
    history: Optional[List[Dict[str, str]]] = None,
    llm: Optional[LLMClient] = None,
    prune_schema: bool = True,
    max_tables: int = 5,
) -> Tuple[str, List[Tuple[DocumentChunk, float]], str]:
    """
    Retrieves relevant schema context and generates a SQLite query.
    Returns: (generated_sql, retrieved_chunks, raw_llm_response).
    (Backward-compatible wrapper around generate_sql_query_full.)
    """
    full = generate_sql_query_full(
        nl_query=nl_query,
        db_path=db_path,
        vector_store=vector_store,
        model_name=model_name,
        top_k=top_k,
        temperature=temperature,
        history=history,
        llm=llm,
        prune_schema=prune_schema,
        max_tables=max_tables,
    )
    return full["sql"], full["chunks"], full["raw_llm_response"]


def auto_repair_sql(
    original_query: str,
    failed_sql: str,
    error_message: str,
    db_path: str,
    model_name: str = "llama3.1",
    llm: Optional[LLMClient] = None,
    prior_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """
    Self-healing mechanism: feeds failed SQL and SQLite error back to LLM to produce a corrected query.
    Prior failed attempts are included so the model doesn't repeat the same mistake.
    """
    llm = _resolve_llm(llm, model_name)

    schema_map = get_full_database_schema(db_path)
    schema_text = format_schema_for_llm(schema_map, include_samples=True)

    prior_text = ""
    if prior_attempts:
        lines = []
        for a in prior_attempts[-3:]:
            lines.append(f"- Attempt SQL: {a.get('sql')}\n  Result: {a.get('error')}")
        prior_text = "\n=== PREVIOUS FAILED REPAIR ATTEMPTS (do NOT repeat these) ===\n" + "\n".join(lines)

    repair_prompt = f"""=== DATABASE SCHEMA ===
{schema_text}

=== ORIGINAL USER REQUEST ===
{original_query}

=== FAILED SQL QUERY ===
{failed_sql}

=== SQLITE ERROR MESSAGE ===
{error_message}
{prior_text}

Fix the error in the SQL query so it runs successfully on SQLite. Return ONLY the corrected SQL in a ```sql ... ``` block."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": repair_prompt},
    ]

    try:
        raw_output = llm.chat(messages, temperature=0.0)
        repaired_sql = extract_sql_from_response(raw_output)
        return repaired_sql, raw_output
    except Exception as e:
        return failed_sql, f"Repair failed: {str(e)}"


def repair_sql(
    original_query: str,
    failed_sql: str,
    error_message: str,
    db_path: str,
    model_name: str = "llama3.1",
    llm: Optional[LLMClient] = None,
    max_attempts: int = 3,
) -> Tuple[str, pd.DataFrame, str, List[Dict[str, Any]]]:
    """
    Retry loop around auto-repair: each candidate is structurally validated
    with sqlglot *before* execution, and the attempt history feeds back into
    the next repair prompt.

    Returns (final_sql, dataframe, status_message, attempts_log).
    A status starting with ✅ or ℹ️ counts as success.
    """
    llm = _resolve_llm(llm, model_name)
    schema_map = get_full_database_schema(db_path)

    attempts: List[Dict[str, Any]] = []
    current_sql, current_error = failed_sql, error_message
    last_df, last_msg = pd.DataFrame(), error_message

    for i in range(1, max_attempts + 1):
        repaired, _raw = auto_repair_sql(
            original_query, current_sql, current_error, db_path,
            llm=llm, prior_attempts=attempts,
        )

        ok, struct_errors = validate_sql_structure(repaired, schema_map)
        if not ok:
            current_error = "Structure check failed: " + "; ".join(struct_errors)
            attempts.append({"attempt": i, "sql": repaired, "error": current_error})
            current_sql = repaired
            last_msg = f"❌ Repair attempt {i} failed structural validation: {'; '.join(struct_errors)}"
            continue

        df, msg, _lat = execute_safe_query(db_path, repaired)
        succeeded = msg.startswith("✅") or msg.startswith("ℹ️")
        attempts.append({"attempt": i, "sql": repaired, "error": None if succeeded else msg})
        if succeeded:
            return repaired, df, msg, attempts

        current_sql, current_error, last_df, last_msg = repaired, msg, df, msg

    return current_sql, last_df, last_msg, attempts


def summarize_results_nl(
    user_query: str,
    sql: str,
    df: pd.DataFrame,
    model_name: str = "llama3.1",
    llm: Optional[LLMClient] = None,
) -> str:
    """
    Generates a concise natural language summary of the query results.
    """
    if df.empty:
        return "The query executed successfully but found 0 matching records."

    llm = _resolve_llm(llm, model_name)

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
        summary = llm.chat(messages, temperature=0.2)
        return summary.strip()
    except Exception as e:
        return f"Summary unavailable ({str(e)})"
