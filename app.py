"""
SQL RAG — Enterprise Natural Language to SQL with Retrieval-Augmented Generation
Powered by Local LLMs (Ollama) & Hybrid Vector Embeddings
"""

import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Any, Optional

from core.database import (
    init_sample_database,
    get_database_tables,
    get_table_details,
    get_full_database_schema,
    format_schema_for_llm,
    execute_safe_query,
    load_csv_to_sqlite,
    validate_sql_structure,
    explain_query_plan,
    DEFAULT_DB_PATH
)
from core.vector_store import (
    VectorStore,
    get_or_build_index,
)
from core.llm import make_llm
from core.rag_engine import (
    generate_sql_query,
    generate_sql_query_full,
    auto_repair_sql,
    repair_sql,
    summarize_results_nl
)
from core.visualizer import (
    classify_columns,
    recommend_chart_type,
    render_plotly_chart
)


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="SQL RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-success { background-color: #DEF7EC; color: #03543F; }
    .badge-info { background-color: #E1EFFE; color: #1E429F; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Management ---
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "last_sql" not in st.session_state:
    st.session_state.last_sql = ""
if "last_df" not in st.session_state:
    st.session_state.last_df = None
if "last_status" not in st.session_state:
    st.session_state.last_status = ""
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""
if "db_path" not in st.session_state:
    st.session_state.db_path = DEFAULT_DB_PATH
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_tables_used" not in st.session_state:
    st.session_state.last_tables_used = []
if "last_rewritten" not in st.session_state:
    st.session_state.last_rewritten = ""
if "last_repair_attempts" not in st.session_state:
    st.session_state.last_repair_attempts = []


# --- Helper to list installed Ollama models ---
def get_installed_ollama_models() -> List[str]:
    try:
        import ollama
        res = ollama.list()
        models = []
        models_list = getattr(res, "models", None) or res.get("models", [])
        for m in models_list:
            name = getattr(m, "model", None) or m.get("name") or m.get("model")
            if name:
                models.append(name)
        return models if models else ["llama3.1:latest"]
    except Exception:
        return ["llama3.1:latest"]


# --- Cached Vector Store Resource ---
@st.cache_resource(show_spinner=False)
def get_cached_vector_store(db_path: str, backend: str, ollama_embed_model: str) -> VectorStore:
    """Builds the vector store, or loads it from the disk cache when the DB is unchanged."""
    return get_or_build_index(db_path, backend=backend, ollama_embed_model=ollama_embed_model)


# Initialize sample DB on start
init_sample_database(DEFAULT_DB_PATH)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    # LLM Provider Selection
    llm_provider = st.radio(
        "LLM Provider",
        options=["ollama", "cloud"],
        format_func=lambda x: "Ollama (local, private)" if x == "ollama" else "Cloud (OpenAI-compatible)",
        help="Local Ollama, or any OpenAI-compatible chat API (e.g. Gemini) for hosted demos",
    )

    cloud_base_url = ""
    cloud_api_key = ""
    if llm_provider == "ollama":
        # Model Selection
        ollama_models = get_installed_ollama_models()
        default_model_idx = 0
        for idx, m in enumerate(ollama_models):
            if "llama3" in m.lower():
                default_model_idx = idx
                break

        selected_model = st.selectbox(
            "LLM Model (Ollama)",
            options=ollama_models,
            index=default_model_idx,
            help="Local LLM model to generate SQL queries"
        )
    else:
        cloud_base_url = st.text_input(
            "API Base URL",
            value="https://generativelanguage.googleapis.com/v1beta/openai/",
            help="Any OpenAI-compatible /chat/completions endpoint",
        )
        cloud_api_key = st.text_input(
            "API Key",
            type="password",
            help="Kept in memory only. On Streamlit Cloud, prefer st.secrets['LLM_API_KEY'].",
        )
        selected_model = st.text_input(
            "Model",
            value="gemini-flash-latest",
            help="Model name at the cloud endpoint",
        )

    # Embedding Backend Selection
    embedding_backend = st.selectbox(
        "Embedding Provider",
        options=["auto", "ollama", "sentence-transformers", "tfidf"],
        index=0,
        format_func=lambda x: {
            "auto": "Auto-Detect (Best Available)",
            "ollama": "Ollama (nomic-embed-text)",
            "sentence-transformers": "SentenceTransformers (BGE/MiniLM)",
            "tfidf": "TF-IDF (Offline, 0-Latency)"
        }.get(x, x),
        help="Backend used to semantically index database tables, columns, and sample values"
    )

    ollama_embed_model = "nomic-embed-text"
    if embedding_backend == "ollama":
        ollama_embed_model = st.text_input("Ollama Embed Model", value="nomic-embed-text")

    # Hyperparameters
    col_k, col_temp = st.columns(2)
    with col_k:
        top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=5)
    with col_temp:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    st.markdown("---")
    st.markdown("### 🗄️ Database Source")

    db_source_type = st.radio(
        "Choose Database",
        options=["Sample Enterprise DB", "Upload Custom SQLite", "Upload CSV File", "Custom File Path"],
        index=0
    )

    if db_source_type == "Sample Enterprise DB":
        st.session_state.db_path = DEFAULT_DB_PATH
        if st.button("🔄 Reset Sample Database", use_container_width=True):
            init_sample_database(DEFAULT_DB_PATH, force_recreate=True)
            st.cache_resource.clear()
            st.success("Sample database reset!")
            st.rerun()

    elif db_source_type == "Upload Custom SQLite":
        uploaded_db = st.file_uploader("Upload .db or .sqlite", type=["db", "sqlite", "sqlite3"])
        if uploaded_db is not None:
            save_path = os.path.join("data", uploaded_db.name)
            os.makedirs("data", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_db.getbuffer())
            st.session_state.db_path = save_path
            st.success(f"Loaded: {uploaded_db.name}")

    elif db_source_type == "Upload CSV File":
        uploaded_csv = st.file_uploader("Upload .csv", type=["csv"])
        if uploaded_csv is not None:
            table_name = st.text_input("Target Table Name", value="dataset")
            if st.button("📥 Import CSV to SQLite", use_container_width=True):
                custom_db_path = "data/custom_csv.db"
                load_csv_to_sqlite(uploaded_csv, table_name=table_name, db_path=custom_db_path)
                st.session_state.db_path = custom_db_path
                st.cache_resource.clear()
                st.success(f"Imported `{uploaded_csv.name}` as `{table_name}` table!")
                st.rerun()

    elif db_source_type == "Custom File Path":
        custom_path = st.text_input("SQLite DB Path", value=st.session_state.db_path)
        if st.button("Connect Path"):
            st.session_state.db_path = custom_path
            st.cache_resource.clear()
            st.rerun()

    # Active DB Schema Explorer in Sidebar
    st.markdown("---")
    st.markdown("### 📋 Schema Explorer")
    if os.path.exists(st.session_state.db_path):
        tables = get_database_tables(st.session_state.db_path)
        st.caption(f"Connected: `{os.path.basename(st.session_state.db_path)}` ({len(tables)} tables)")
        for t in tables:
            details = get_table_details(st.session_state.db_path, t)
            with st.expander(f"📦 {t} ({details['row_count']} rows)"):
                col_df = pd.DataFrame([
                    {"Column": c["name"], "Type": c["type"], "PK": "🔑" if c["is_pk"] else ""}
                    for c in details["columns"]
                ])
                st.dataframe(col_df, hide_index=True, use_container_width=True)
                if details["foreign_keys"]:
                    st.caption("Foreign Keys:")
                    for fk in details["foreign_keys"]:
                        st.code(f"{fk['from_column']} ➔ {fk['to_table']}.{fk['to_column']}", language="sql")
    else:
        st.warning(f"Database not found at `{st.session_state.db_path}`")


# --- LLM client factory (Streamlit-secrets aware) ---
def _secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_llm_client():
    """Builds the LLM client for the provider chosen in the sidebar."""
    if llm_provider == "cloud":
        return make_llm(
            provider="cloud",
            model=selected_model or _secret("LLM_MODEL", "gemini-flash-latest"),
            base_url=cloud_base_url or _secret("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            api_key=cloud_api_key or _secret("LLM_API_KEY", ""),
        )
    return make_llm(provider="ollama", model=selected_model)


# --- Main Application Area ---
st.markdown('<div class="main-header">⚡ SQL RAG Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Convert natural language questions into accurate SQL, inspect schemas with semantic vector search, and visualize data instantly.</div>', unsafe_allow_html=True)

# Build or retrieve vector store
current_db = st.session_state.db_path
if not os.path.exists(current_db):
    st.error(f"Cannot locate database at `{current_db}`. Please select a valid database in the sidebar.")
    st.stop()

with st.spinner("Indexing database schema with vector embeddings..."):
    vector_store = get_cached_vector_store(current_db, embedding_backend, ollama_embed_model)

# Active backend indicator
st.markdown(
    f'<span class="metric-badge badge-info">LLM: {selected_model}</span>'
    f'<span class="metric-badge badge-success">Embedding: {vector_store.active_backend.upper()}</span>'
    f'<span class="metric-badge badge-info">Database: {os.path.basename(current_db)}</span>',
    unsafe_allow_html=True
)
st.write("")

# Quick-click example questions for the sample DB
if current_db == DEFAULT_DB_PATH:
    st.markdown("**💡 Quick Examples:**")
    example_cols = st.columns(4)
    examples = [
        "Top 5 highest paid employees and their departments",
        "Total sales revenue and units sold per region",
        "List all in-progress projects with budget > $100,000",
        "Average employee performance rating by city"
    ]
    for i, ex in enumerate(examples):
        with example_cols[i % 4]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.current_query = ex
                st.rerun()

# NL Query Input Form
with st.form("query_form", clear_on_submit=False):
    user_query = st.text_area(
        "💬 Ask a question in plain English:",
        value=st.session_state.current_query,
        placeholder="e.g. 'Show the total sales revenue by product category ordered by revenue descending'",
        height=90
    )
    col_submit, col_summary, col_repair, col_follow = st.columns([2, 2, 2, 2])
    with col_submit:
        submitted = st.form_submit_button("🚀 Generate & Execute SQL", type="primary", use_container_width=True)
    with col_summary:
        generate_summary = st.checkbox("Generate Natural Language Summary", value=True)
    with col_repair:
        enable_auto_repair = st.checkbox("Auto-repair SQL on error", value=True)
    with col_follow:
        enable_followups = st.checkbox(
            "Conversational follow-ups",
            value=True,
            help="Resolve follow-up questions ('now only 2023') against chat history",
        )

# Process Query
if submitted and user_query.strip():
    st.session_state.current_query = user_query.strip()
    llm = get_llm_client()
    history = st.session_state.chat_history if enable_followups else None

    with st.spinner("🤖 Retrieving schema context & generating SQL query..."):
        t0 = time.perf_counter()
        full = generate_sql_query_full(
            nl_query=user_query.strip(),
            db_path=current_db,
            vector_store=vector_store,
            top_k=top_k,
            temperature=temperature,
            history=history,
            llm=llm,
        )
        gen_time_ms = (time.perf_counter() - t0) * 1000.0

    sql = full["sql"]
    st.session_state.last_sql = sql
    st.session_state.last_retrieved = full["chunks"]
    st.session_state.last_tables_used = full["tables_used"]
    st.session_state.last_rewritten = full["rewritten_question"]
    st.session_state.last_repair_attempts = []
    repair_attempts: list = []

    if not sql:
        st.error(f"Failed to generate SQL: {full['raw_llm_response']}")
    else:
        # Structural pre-check: catch hallucinated tables/columns before execution
        schema_map = get_full_database_schema(current_db)
        struct_ok, struct_errors = validate_sql_structure(sql, schema_map)
        if not struct_ok:
            df = pd.DataFrame()
            status_msg = "❌ Structure check failed: " + "; ".join(struct_errors)
            exec_latency = 0.0
        else:
            with st.spinner("⚡ Executing SQL query on database..."):
                df, status_msg, exec_latency = execute_safe_query(current_db, sql)

        # Auto-repair retry loop if execution failed
        if (status_msg.startswith("❌") or status_msg.startswith("⚠️")) and enable_auto_repair:
            st.warning(f"Initial query failed: {status_msg}. Attempting AI auto-repair...")
            sql, df, status_msg, repair_attempts = repair_sql(
                original_query=user_query.strip(),
                failed_sql=sql,
                error_message=status_msg,
                db_path=current_db,
                llm=llm,
                max_attempts=3,
            )
            st.session_state.last_sql = sql
            st.session_state.last_repair_attempts = repair_attempts

        st.session_state.last_df = df
        st.session_state.last_status = status_msg

        # Generate NL summary if requested and results are present
        if generate_summary and not df.empty:
            with st.spinner("📝 Generating business intelligence summary..."):
                summary = summarize_results_nl(
                    user_query=user_query.strip(),
                    sql=st.session_state.last_sql,
                    df=df,
                    llm=llm,
                )
                st.session_state.last_summary = summary
        else:
            st.session_state.last_summary = ""

        # Record to query history
        st.session_state.query_history.insert(0, {
            "timestamp": time.strftime("%H:%M:%S"),
            "query": user_query.strip(),
            "sql": st.session_state.last_sql,
            "rows": len(df),
            "status": "Success" if (status_msg.startswith("✅") or status_msg.startswith("ℹ️")) else "Error",
            "gen_time_ms": gen_time_ms,
            "exec_time_ms": exec_latency,
            "repairs": len(repair_attempts),
        })

        # Conversational memory for follow-up resolution
        if enable_followups:
            st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"Ran SQL: {st.session_state.last_sql}"}
            )


# --- Results Section ---
if st.session_state.last_sql:
    st.markdown("---")

    # Status Banner
    if "✅" in st.session_state.last_status:
        st.success(st.session_state.last_status)
    elif "⚠️" in st.session_state.last_status:
        st.warning(st.session_state.last_status)
    else:
        st.error(st.session_state.last_status)

    # Follow-up rewrite + pruned-tables context
    if st.session_state.last_rewritten and st.session_state.last_rewritten != st.session_state.current_query:
        st.caption(f"🔁 Follow-up interpreted as: “{st.session_state.last_rewritten}”")
    if st.session_state.last_tables_used:
        badges = " ".join(
            f'<span class="metric-badge badge-info">📦 {t}</span>'
            for t in st.session_state.last_tables_used
        )
        st.markdown(f"Tables used: {badges}", unsafe_allow_html=True)

    # Optional NL Summary Box
    if st.session_state.last_summary:
        st.info(f"💡 **Key Insight:** {st.session_state.last_summary}")

    # Main Tabs
    tab_results, tab_sql, tab_rag, tab_history = st.tabs([
        "📊 Results & Visualizations",
        "💻 SQL Query & Editor",
        "🧠 RAG Insights & Diagnostics",
        "📜 Query History"
    ])

    # --- TAB 1: Results & Charts ---
    with tab_results:
        df = st.session_state.last_df
        if df is not None and not df.empty:
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Total Rows", len(df))
            with col_m2:
                st.metric("Total Columns", len(df.columns))
            with col_m3:
                num_cols = classify_columns(df)["numeric"]
                st.metric("Numeric Metrics", len(num_cols))

            # Interactive DataFrame
            st.dataframe(df, use_container_width=True)

            # Export Buttons
            exp_col1, exp_col2 = st.columns([1, 1])
            with exp_col1:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv_bytes, file_name="sql_results.csv", mime="text/csv", use_container_width=True)
            with exp_col2:
                json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
                st.download_button("📥 Download JSON", data=json_bytes, file_name="sql_results.json", mime="application/json", use_container_width=True)

            # Visualization Section
            if len(df.columns) >= 1:
                st.markdown("#### 📈 Interactive Visualizer")
                rec_chart, rec_x, rec_y = recommend_chart_type(df)

                vcol1, vcol2, vcol3, vcol4 = st.columns(4)
                with vcol1:
                    chart_type = st.selectbox(
                        "Chart Type",
                        options=["Bar", "Line", "Area", "Pie", "Donut", "Scatter", "Histogram"],
                        index=["Bar", "Line", "Area", "Pie", "Donut", "Scatter", "Histogram"].index(rec_chart.capitalize()) if rec_chart.capitalize() in ["Bar", "Line", "Area", "Pie", "Donut", "Scatter", "Histogram"] else 0
                    )
                with vcol2:
                    x_axis = st.selectbox("X-Axis", options=df.columns, index=list(df.columns).index(rec_x) if rec_x in df.columns else 0)
                with vcol3:
                    y_options = [None] + list(df.columns)
                    y_default_idx = (y_options.index(rec_y)) if rec_y in df.columns else 0
                    y_axis = st.selectbox("Y-Axis", options=y_options, index=y_default_idx)
                with vcol4:
                    color_options = [None] + list(df.columns)
                    color_axis = st.selectbox("Color / Group", options=color_options, index=0)

                fig = render_plotly_chart(
                    df=df,
                    chart_type=chart_type,
                    x_col=x_axis,
                    y_col=y_axis,
                    color_col=color_axis
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Select appropriate numeric columns for this chart type.")
        else:
            st.info("No data returned by the query.")

    # --- TAB 2: SQL Query & In-Place Editor ---
    with tab_sql:
        st.markdown("#### Generated SQL Query")
        st.code(st.session_state.last_sql, language="sql")

        with st.expander("🧭 Query Plan (EXPLAIN QUERY PLAN)"):
            st.caption("How SQLite will execute this query — useful for spotting full table scans.")
            st.code(explain_query_plan(current_db, st.session_state.last_sql), language="text")

        st.markdown("#### ✏️ Interactive SQL Editor & Runner")
        st.caption("You can modify the SQL below and re-run it directly against the active database:")
        edited_sql = st.text_area("Edit SQL:", value=st.session_state.last_sql, height=130, key="edited_sql_box")

        if st.button("▶️ Execute Edited SQL", type="secondary"):
            re_df, re_status, re_latency = execute_safe_query(current_db, edited_sql)
            st.session_state.last_sql = edited_sql
            st.session_state.last_df = re_df
            st.session_state.last_status = re_status
            st.rerun()

    # --- TAB 3: RAG Insights & Diagnostics ---
    with tab_rag:
        st.markdown("#### 🔍 Retrieved Schema Chunks (RAG Context)")
        if st.session_state.last_retrieved:
            for i, (chunk, score) in enumerate(st.session_state.last_retrieved):
                with st.expander(f"Chunk #{i+1} — {chunk.metadata.get('type', 'Context')} (Relevance: {score:.3f})", expanded=(i == 0)):
                    st.write(f"**Content:** {chunk.text}")
                    st.json(chunk.metadata)
        else:
            st.write("No vector chunks retrieved.")

        st.markdown("#### 📋 Schema Sent to the LLM (pruned to relevant tables)")
        with st.expander("View Schema Prompt"):
            schema_map = get_full_database_schema(current_db)
            st.text(format_schema_for_llm(schema_map))

        if st.session_state.last_repair_attempts:
            st.markdown("#### 🔧 Auto-Repair Attempts")
            for a in st.session_state.last_repair_attempts:
                status = "✅ fixed" if a["error"] is None else "❌ failed"
                with st.expander(f"Attempt {a['attempt']} — {status}"):
                    st.code(a["sql"], language="sql")
                    if a["error"]:
                        st.caption(a["error"])

    # --- TAB 4: Query History ---
    with tab_history:
        st.markdown("#### 📜 Execution History")
        if st.session_state.query_history:
            for item in st.session_state.query_history:
                h_col1, h_col2, h_col3 = st.columns([4, 1, 1])
                with h_col1:
                    st.write(f"**[{item['timestamp']}]** `{item['query']}`")
                    st.code(item['sql'], language="sql")
                with h_col2:
                    st.caption(f"Rows: {item['rows']}\nStatus: {item['status']}")
                with h_col3:
                    st.caption(f"Gen: {item['gen_time_ms']:.0f}ms\nExec: {item['exec_time_ms']:.1f}ms")
                st.markdown("---")
        else:
            st.write("No queries run yet.")
