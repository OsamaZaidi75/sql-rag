"""
Execution-accuracy evals for SQL RAG against a golden question set.

Metric: execution match — the generated SQL is correct iff executing it
returns the same multiset of rows as the golden answer (order-insensitive,
floats rounded). Comparing result sets instead of SQL strings is the honest
metric: many different queries can be equally correct.

Modes:
  --mock    No LLM needed. A deterministic mock returns the golden SQL, so this
            verifies the harness itself end-to-end (expect 25/25).
  (default) Real run through generate_sql_query with Ollama (or --provider cloud).

Run:  python -m evals.run_evals --mock
      python -m evals.run_evals --model llama3.1 --top-k 5
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.database import init_sample_database, execute_safe_query, DEFAULT_DB_PATH
from core.vector_store import VectorStore, index_database_schema
from core.rag_engine import generate_sql_query
from core.llm import make_llm

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden.json")


def normalize(value):
    if isinstance(value, float):
        return round(value, 4)
    return value


def rows_to_counter(rows):
    """Order-insensitive multiset of result rows."""
    return Counter(tuple(normalize(v) for v in row) for row in rows)


def df_to_counter(df):
    return rows_to_counter(df.itertuples(index=False, name=None))


class MockLLM:
    """Deterministic stand-in: returns the golden SQL for the asked question."""

    label = "mock"

    def __init__(self, golden):
        self.golden = golden

    def chat(self, messages, temperature=0.0, max_tokens=None):
        prompt = messages[-1]["content"]
        for item in self.golden:
            if item["question"] in prompt:
                return f"```sql\n{item['expected_sql']}\n```"
        return "```sql\nSELECT 1\n```"


def main() -> int:
    parser = argparse.ArgumentParser(description="SQL RAG execution-accuracy evals")
    parser.add_argument("--mock", action="store_true", help="use deterministic mock LLM (harness self-test)")
    parser.add_argument("--model", default="llama3.1", help="Ollama model for real runs")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "cloud"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    init_sample_database(args.db)
    vstore = VectorStore(backend="tfidf")  # deterministic, no downloads
    index_database_schema(args.db, vstore)

    if args.mock:
        llm = MockLLM(golden)
    else:
        llm = make_llm(provider=args.provider, model=args.model,
                       base_url=args.base_url, api_key=args.api_key)

    passed = 0
    print(f"{'id':28s} {'rows':>6s}  result")
    print("-" * 48)
    for item in golden:
        qid, question = item["id"], item["question"]
        expected = rows_to_counter(item["expected_rows"])
        try:
            sql, _chunks, _raw = generate_sql_query(
                nl_query=question, db_path=args.db, vector_store=vstore,
                top_k=args.top_k, llm=llm,
            )
            df, msg, _lat = execute_safe_query(args.db, sql)
            ok = df_to_counter(df) == expected and (msg.startswith("✅") or msg.startswith("ℹ️"))
            detail = f"{len(df)} rows"
        except Exception as e:
            ok, detail = False, f"error: {str(e)[:60]}"
        passed += ok
        print(f"{qid:28s} {detail:>6s}  {'PASS' if ok else 'FAIL'}")

    n = len(golden)
    acc = 100.0 * passed / n if n else 0.0
    print("-" * 48)
    print(f"Execution accuracy: {passed}/{n} ({acc:.1f}%)  [llm={llm.label}]")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
