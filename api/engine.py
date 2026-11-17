"""Engine: wires the existing ``core/`` RAG pipeline to the API.

Nothing here re-implements retrieval, safety, or repair — it calls
``core.rag_engine`` / ``core.database`` / ``core.vector_store`` exactly as
the Streamlit app does, and shapes the results as JSON.

The LLM is injectable (``engine.llm_override``) so tests and the mock-eval
path can run without any provider. In production the client comes from
:func:`api.config.load_settings` — NVIDIA API by default when
``NVIDIA_API_KEY`` is set.

NVIDIA note: the default model (Nemotron 3 Ultra) is a reasoning model —
its thinking lands in a separate ``reasoning_content`` field and consumes
completion tokens *first*, so a small ``max_tokens`` can return no content
at all. :class:`BoundedLLM` therefore enforces a generous cap on every
cloud call (2048 by default, never below 1024). The answer is always read
from ``choices[0].message.content`` only (see ``core/llm.py``);
``reasoning_content`` is ignored and never surfaced.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from core.database import (
    DEFAULT_DB_PATH,
    execute_safe_query,
    explain_query_plan,
    get_database_tables,
    get_full_database_schema,
    get_table_details,
    init_sample_database,
    validate_sql_structure,
)
from core.llm import LLMClient, make_llm
from core.rag_engine import (
    generate_sql_query,
    generate_sql_query_full,
    repair_sql,
    summarize_results_nl,
)
from core.vector_store import VectorStore, get_or_build_index

from .config import Settings, load_settings

# Low temperature: SQL generation wants determinism, not creativity.
DEFAULT_TEMPERATURE = 0.2

# Reasoning models (Nemotron) spend completion tokens on thinking before
# they answer; a small cap can return no content at all. SQL calls get a
# generous cap, and no cloud call is ever capped below the floor.
SQL_MAX_TOKENS = 2048
MIN_CLOUD_MAX_TOKENS = 1024


class BoundedLLM:
    """Delegating wrapper enforcing a generous ``max_tokens`` on cloud calls.

    The core pipeline calls ``llm.chat(...)`` without a cap (SQL generation,
    summaries) or with a small one (follow-up rewrites, 150 tokens). Against
    a reasoning model both are unsafe: thinking consumes completion tokens
    first, so the cap must leave room for reasoning *plus* the answer. The
    model still stops at end-of-sequence, so a generous cap costs nothing
    when the answer is short. Ollama calls pass through untouched.
    """

    def __init__(self, inner: LLMClient):
        self._inner = inner

    @property
    def label(self) -> str:
        return self._inner.label

    @property
    def config(self):  # expose the wrapped config for introspection
        return self._inner.config

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self._inner.config.provider == "cloud":
            cap = SQL_MAX_TOKENS if max_tokens is None else max_tokens
            max_tokens = max(cap, MIN_CLOUD_MAX_TOKENS)
        return self._inner.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "evals", "golden.json"
)


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame rows as JSON-safe records (NaN -> null)."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def chart_recommendation(df: pd.DataFrame) -> Dict[str, Any]:
    """Best chart for a result set, reusing the core recommender."""
    if df is None or df.empty:
        return {"type": "table", "x": None, "y": None}
    try:
        from core.visualizer import recommend_chart_type

        chart, x, y = recommend_chart_type(df)
        return {"type": chart, "x": x, "y": y}
    except Exception:
        return {"type": "table", "x": None, "y": None}


def load_golden() -> List[Dict[str, Any]]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


class Engine:
    """Holds the DB, the vector index, and the LLM client."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()
        self._store: Optional[VectorStore] = None
        self._store_lock = threading.Lock()
        self._llm: Optional[LLMClient] = None
        self.llm_override: Optional[LLMClient] = None
        self._eval_runs: Dict[str, Dict[str, Any]] = {}
        self._eval_lock = threading.Lock()

    # -- setup ----------------------------------------------------------

    def ensure_db(self) -> str:
        """Create the sample DB if it is missing; never overwrite it."""
        return init_sample_database(self.settings.db_path)

    @property
    def vector_store(self) -> VectorStore:
        if self._store is None:
            with self._store_lock:
                if self._store is None:
                    self.ensure_db()
                    self._store = get_or_build_index(
                        self.settings.db_path,
                        backend=self.settings.embedding_backend,
                        ollama_embed_model=self.settings.ollama_embed_model,
                    )
        return self._store

    @property
    def llm(self) -> LLMClient:
        if self.llm_override is not None:
            return self.llm_override
        if self._llm is None:
            s = self.settings
            self._llm = BoundedLLM(
                make_llm(
                    provider=s.llm_provider,
                    model=s.llm_model,
                    base_url=s.llm_base_url,
                    api_key=s.llm_api_key,
                    timeout_secs=s.llm_timeout_secs,
                )
            )
        return self._llm

    def llm_status(self) -> Dict[str, Any]:
        """Availability probe. Cloud keys are never verified by a live call
        (that would spend a request); presence of the key is what counts."""
        s = self.settings
        if self.llm_override is not None:
            return {
                "available": True,
                "label": getattr(self.llm_override, "label", "override"),
                "detail": "injected LLM (test/mock)",
            }
        if s.llm_provider == "cloud":
            return {
                "available": bool(s.llm_api_key),
                "label": s.llm_label,
                "detail": (
                    "API key configured"
                    if s.llm_api_key
                    else "no API key in environment (set NVIDIA_API_KEY)"
                ),
            }
        try:
            import ollama

            ollama.list()
            return {"available": True, "label": s.llm_label,
                    "detail": "Ollama server reachable"}
        except Exception as e:
            return {"available": False, "label": s.llm_label,
                    "detail": f"Ollama not reachable ({e})"}

    # -- schema ---------------------------------------------------------

    def schema(self) -> Dict[str, Any]:
        self.ensure_db()
        schema_map = get_full_database_schema(self.settings.db_path)
        tables = []
        for name, details in schema_map.items():
            tables.append(
                {
                    "name": name,
                    "row_count": details.get("row_count", 0),
                    "columns": details.get("columns", []),
                    "foreign_keys": details.get("foreign_keys", []),
                }
            )
        return {
            "database": os.path.basename(self.settings.db_path),
            "db_path": self.settings.db_path,
            "table_count": len(tables),
            "tables": tables,
        }

    def table_details(self, name: str) -> Dict[str, Any]:
        self.ensure_db()
        # get_table_details returns a (column-less) dict even for missing
        # tables, so check membership first for a truthful 404.
        if name not in get_database_tables(self.settings.db_path):
            return {}
        return get_table_details(self.settings.db_path, name)

    # -- ask ------------------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = DEFAULT_TEMPERATURE,
        auto_repair: bool = True,
        summarize: bool = True,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: rewrite -> retrieve -> generate -> validate ->
        execute -> (repair) -> summarize. Mirrors app.py's flow."""
        db_path = self.ensure_db()
        llm = self.llm

        t0 = time.perf_counter()
        full = generate_sql_query_full(
            nl_query=question,
            db_path=db_path,
            vector_store=self.vector_store,
            top_k=top_k,
            temperature=temperature,
            history=history,
            llm=llm,
        )
        gen_time_ms = (time.perf_counter() - t0) * 1000.0

        sql = full["sql"]
        repair_attempts: List[Dict[str, Any]] = []
        summary = ""
        exec_latency = 0.0

        if not sql:
            df = pd.DataFrame()
            status = f"Failed to generate SQL: {full['raw_llm_response']}"
        else:
            schema_map = get_full_database_schema(db_path)
            struct_ok, struct_errors = validate_sql_structure(sql, schema_map)
            if not struct_ok:
                df = pd.DataFrame()
                status = "Structure check failed: " + "; ".join(struct_errors)
            else:
                df, status, exec_latency = execute_safe_query(
                    db_path,
                    sql,
                    max_rows=self.settings.max_rows,
                    timeout_secs=self.settings.query_timeout_secs,
                )

            if (status.startswith("❌") or status.startswith("⚠️")) and auto_repair:
                sql, df, status, repair_attempts = repair_sql(
                    original_query=question,
                    failed_sql=sql,
                    error_message=status,
                    db_path=db_path,
                    llm=llm,
                    max_attempts=3,
                )

        ok = status.startswith("✅") or status.startswith("ℹ️")
        if summarize and ok and not df.empty:
            try:
                summary = summarize_results_nl(
                    question, sql, df, llm=llm,
                )
            except Exception as e:  # summary must never fail the answer
                summary = f"Summary unavailable ({e})"

        chunks = [
            {
                "text": chunk.text,
                "type": (chunk.metadata or {}).get("type", "info"),
                "score": round(float(score), 3),
                "metadata": chunk.metadata or {},
            }
            for chunk, score in full["chunks"]
        ]

        return {
            "question": question,
            "rewritten_question": full["rewritten_question"],
            "sql": sql,
            "status": status,
            "ok": ok,
            "columns": list(df.columns) if df is not None else [],
            "rows": df_to_records(df),
            "row_count": len(df) if df is not None else 0,
            "summary": summary,
            "chart": chart_recommendation(df),
            "chunks": chunks,
            "tables_used": full["tables_used"],
            "pruned": full["pruned"],
            "repair_attempts": repair_attempts,
            "gen_time_ms": round(gen_time_ms, 1),
            "exec_time_ms": round(exec_latency, 1),
            "llm": llm.label,
        }

    # -- direct SQL -----------------------------------------------------

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        db_path = self.ensure_db()
        df, status, latency = execute_safe_query(
            db_path,
            sql,
            max_rows=self.settings.max_rows,
            timeout_secs=self.settings.query_timeout_secs,
        )
        return {
            "sql": sql,
            "status": status,
            "ok": status.startswith("✅") or status.startswith("ℹ️"),
            "columns": list(df.columns),
            "rows": df_to_records(df),
            "row_count": len(df),
            "chart": chart_recommendation(df),
            "exec_time_ms": round(latency, 1),
        }

    def explain(self, sql: str) -> Dict[str, Any]:
        db_path = self.ensure_db()
        return {"sql": sql, "plan": explain_query_plan(db_path, sql)}

    # -- evals ----------------------------------------------------------

    def start_eval(self, mock: bool) -> Dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        with self._eval_lock:
            self._eval_runs[run_id] = {
                "run_id": run_id,
                "status": "running",
                "mock": mock,
                "passed": 0,
                "total": 0,
                "accuracy": 0.0,
                "llm": "mock" if mock else self.settings.llm_label,
                "results": [],
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        thread = threading.Thread(
            target=self._run_eval, args=(run_id, mock), daemon=True
        )
        thread.start()
        return {"run_id": run_id, "status": "running", "mock": mock}

    def get_eval(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._eval_lock:
            run = self._eval_runs.get(run_id)
            return dict(run) if run else None

    def _run_eval(self, run_id: str, mock: bool) -> None:
        from evals.run_evals import MockLLM, df_to_counter, rows_to_counter

        try:
            golden = load_golden()
            db_path = self.ensure_db()
            llm = MockLLM(golden) if mock else self.llm
            store = self.vector_store
            passed = 0
            results = []
            for item in golden:
                t0 = time.perf_counter()
                try:
                    sql, _ch, _raw = generate_sql_query(
                        nl_query=item["question"],
                        db_path=db_path,
                        vector_store=store,
                        top_k=5,
                        llm=llm,
                    )
                    df, msg, _lat = execute_safe_query(db_path, sql)
                    ok = (
                        df_to_counter(df) == rows_to_counter(item["expected_rows"])
                        and (msg.startswith("✅") or msg.startswith("ℹ️"))
                    )
                    detail = f"{len(df)} rows"
                except Exception as e:
                    ok, detail, sql = False, f"error: {str(e)[:80]}", ""
                elapsed = time.perf_counter() - t0
                passed += 1 if ok else 0
                results.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "ok": bool(ok),
                        "detail": detail,
                        "sql": sql,
                        "elapsed_s": round(elapsed, 3),
                    }
                )
                with self._eval_lock:
                    run = self._eval_runs[run_id]
                    run["passed"] = passed
                    run["total"] = len(golden)
                    run["results"] = list(results)
            with self._eval_lock:
                run = self._eval_runs[run_id]
                run["status"] = "done"
                run["accuracy"] = (
                    round(100.0 * passed / len(golden), 1) if golden else 0.0
                )
        except Exception as e:
            with self._eval_lock:
                run = self._eval_runs.get(run_id)
                if run:
                    run["status"] = "error"
                    run["error"] = str(e)[:300]


# Process-wide singleton; tests build their own Engine instead.
engine = Engine()
