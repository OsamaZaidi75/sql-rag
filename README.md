# ⚡ SQL RAG Studio — Natural Language to SQL

A high-performance, fully local **Retrieval-Augmented Generation (RAG) SQL Engine** powered by local LLMs (**Ollama**) and hybrid semantic vector indexing.

Translate complex natural language questions into accurate, performant, and safe SQLite queries with instant visual analytics, AI auto-repair, and interactive database exploration.

---

## 🚀 Key Features

- **Local LLM Inference**: 100% private, free local execution via Ollama (`llama3.1`, `qwen`, `deepseek-r1`, `mistral`, etc.).
- **Hybrid Vector Retrieval**: Multi-backend embedding engine with automatic fallback:
  1. **Ollama Embeddings** (`nomic-embed-text`)
  2. **SentenceTransformers** (`BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2`)
  3. **Character N-Gram TF-IDF Vectorizer** (Zero-network, zero-download offline fallback)
- **True RAG Schema Augmentation**: Granularly indexes table schemas, column data types, foreign key relationships, distinct categorical values, and SQLite query patterns into vector memory.
- **Safety Guardrails**: Strict read-only query validator blocking destructive DDL/DML operations (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`).
- **AI Self-Correction / Auto-Repair**: Automatically detects SQLite syntax or operational errors and uses the LLM to self-heal and re-execute queries.
- **Natural Language Summaries**: Generates executive plain-English data insights from query results.
- **Interactive Plotly Visualizations**: Automatic chart type recommendation (Bar, Line, Area, Pie, Donut, Scatter, Histogram) with multi-axis selectors.
- **Multi-Source Database Management**:
  - Built-in Enterprise Sample Database (4 relational tables: `departments`, `employees`, `projects`, `sales_transactions`)
  - Custom SQLite Database file connection (`.db`, `.sqlite`, `.sqlite3`)
  - Instant CSV file ingestion into SQLite
- **In-Place SQL Editor & History**: Inspect, modify, and rerun generated SQL queries on the fly with full session history tracking.

---

## 📦 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: [Download & Install Ollama](https://ollama.com/download)

### 2. Pull Ollama Models
```bash
# Pull LLM for SQL generation
ollama pull llama3.1

# (Optional) Pull embedding model for dense vector search
ollama pull nomic-embed-text
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or run `setup.bat` on Windows)*

### 4. Run Test Suite
```bash
python -m pytest -v
```

### 5. Launch the Studio
```bash
streamlit run app.py
```

---

## 🏗️ Architecture

```
User Question
     │
     ▼
[ Hybrid Vector Store ] ───► Retrieves relevant Tables, Columns,
     │                        Sample Values & Join Patterns
     ▼
[ RAG Engine & Prompt ] ───► Augments Schema Context + SQLite Dialect Rules
     │
     ▼
[ Ollama LLM (llama3.1) ] ──► Generates SQL Query
     │
     ▼
[ Safety & Guardrails ] ────► Validates Read-Only & Disallows DDL/DML
     │
     ▼
[ SQLite Database ] ────────► Executes Query (with Auto-Repair on syntax error)
     │
     ▼
[ Plotly Visualizer & NL Summary ] ──► Dataframe + Interactive Charts + AI Insights
```

---

## 📊 Sample Database Schema

The built-in sample database contains 4 interconnected enterprise tables:
1. `departments`: id, name, budget, location, created_at
2. `employees`: id, name, department_id, role, salary, hire_date, city, email, performance_rating
3. `projects`: id, name, department_id, lead_id, budget, status, start_date, end_date
4. `sales_transactions`: id, employee_id, customer_name, product_category, amount, units_sold, transaction_date, region

---

## 🧪 Testing

The repository includes comprehensive unit and end-to-end integration tests:
```bash
# Run all tests
python -m pytest -v

# Run with stdout logging
python -m pytest -s -v
```

---

## 🛡️ License

MIT License. Free and open source.
