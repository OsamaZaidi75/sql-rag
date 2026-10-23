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
      python -m evals.run_evals --mock --junit reports/junit.xml
      python -m evals.run_evals --mock --fail-under 95   # CI gate: fail unless accuracy >= 95%
                                                     # (prints a JSON verdict line first)
      python -m evals.run_evals --mock --json reports/verdict.json
                                                     # archive verdict + per-question results as JSON
      python -m evals.run_evals --mock --no-history  # local debug: no history row recorded

  An explicitly-passed --db must exist and be readable: a typo'd path fails
  fast (exit 2) instead of silently eval'ing a freshly-initialized sample DB.

Every run also appends a one-row summary to reports/daily/eval_history.csv
(unless --no-history is given), so eval accuracy over time can be charted
without re-running history.
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.database import init_sample_database, execute_safe_query, DEFAULT_DB_PATH
from core.vector_store import VectorStore, index_database_schema
from core.rag_engine import generate_sql_query
from core.llm import make_llm

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden.json")

# Where every eval run appends its one-row summary (date, passed, total,
# accuracy_pct, llm) — a running history of golden-eval scores.
EVAL_HISTORY_PATH = "reports/daily/eval_history.csv"


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


def verdict_dict(passed: int, total: int, fail_under: float | None) -> dict:
    """Machine-readable gate verdict: {"accuracy", "fail_under", "ok"}.

    fail_under=None means "no gate" → ok iff every question passed.
    accuracy is execution accuracy in percent, rounded to 1 decimal.
    """
    acc = 100.0 * passed / total if total else 0.0
    ok = (passed == total) if fail_under is None else acc >= fail_under
    return {"accuracy": round(acc, 1), "fail_under": fail_under, "ok": ok}


def gate_exit(passed: int, total: int, fail_under: float | None) -> int:
    """Map eval results to a process exit code.

    No gate: exit 0 only if every question passed.
    --fail-under PCT: exit 0 iff execution accuracy >= PCT (CI gate); a few
    individual failures are tolerated as long as the threshold is met.

    The gated path also prints a machine-readable JSON verdict line first, so
    CI dashboards can consume the gate result without parsing prose:
    {"accuracy": 96.0, "fail_under": 95, "ok": true}
    """
    if fail_under is None:
        return 0 if passed == total else 1
    acc = 100.0 * passed / total if total else 0.0
    verdict = verdict_dict(passed, total, fail_under)
    print(json.dumps(verdict))
    if not verdict["ok"]:
        print(f"FAIL: accuracy {acc:.1f}% is below --fail-under {fail_under}%")
        return 1
    print(f"OK: accuracy {acc:.1f}% meets --fail-under {fail_under}%")
    return 0


def record_eval_history(date_str: str, passed: int, total: int,
                        accuracy: float, llm: str | None = None,
                        path: str = EVAL_HISTORY_PATH) -> None:
    """Append one row for this eval run to the history CSV.

    Writes the header ``date,passed,total,accuracy_pct,llm`` when the file
    does not exist yet. Parent directories are created as needed.
    """
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["date", "passed", "total", "accuracy_pct", "llm"])
        writer.writerow([date_str, passed, total, round(accuracy, 1), llm or ""])


def parse_args(argv=None):
    """CLI parser for run_evals; factored out so tests can probe the flags."""
    parser = argparse.ArgumentParser(description="SQL RAG execution-accuracy evals")
    parser.add_argument("--mock", action="store_true", help="use deterministic mock LLM (harness self-test)")
    parser.add_argument("--model", default="llama3.1", help="Ollama model for real runs")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "cloud"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", default=DEFAULT_DB_PATH,
                        help="SQLite database to eval against (must exist and be "
                             "readable when passed explicitly; the default is "
                             "auto-initialized as the sample DB)")
    parser.add_argument("--junit", default=None, metavar="PATH",
                        help="write JUnit XML report to PATH for CI integration")
    parser.add_argument("--fail-under", type=float, default=None, metavar="PCT",
                        help="CI gate: exit non-zero unless execution accuracy is "
                             "at least PCT%% (e.g. --fail-under 95)")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="write the gate verdict (accuracy, fail_under, ok) plus "
                             "per-question results as JSON to PATH, for CI build artifacts")
    parser.add_argument("--no-history", action="store_true", default=False,
                        help="skip recording this run in reports/daily/eval_history.csv "
                             "(local debugging; CI/workflow runs should record)")
    args = parser.parse_args(argv)
    if args.fail_under is not None and not 0 <= args.fail_under <= 100:
        parser.error(f"--fail-under must be between 0 and 100 (got {args.fail_under})")
    if args.top_k <= 0:
        parser.error(f"--top-k must be a positive integer (got {args.top_k})")
    # An explicitly-passed --db must point at a real, readable database file.
    # main() auto-initializes the *default* DB on a fresh checkout, so the
    # default is exempt: only a deliberate --db with a typo'd path fails fast
    # instead of silently eval'ing a freshly-initialized sample DB.
    raw = sys.argv[1:] if argv is None else argv
    if any(tok == "--db" or tok.startswith("--db=") for tok in raw):
        if not os.path.isfile(args.db):
            parser.error(f"--db '{args.db}' does not exist or is not a file; "
                         "pass the path to an existing SQLite database")
        elif not os.access(args.db, os.R_OK):
            parser.error(f"--db '{args.db}' is not readable")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

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
    cases = []  # (qid, ok, detail, elapsed_s) for the JUnit report
    print(f"{'id':28s} {'rows':>6s}  result")
    print("-" * 48)
    for item in golden:
        qid, question = item["id"], item["question"]
        expected = rows_to_counter(item["expected_rows"])
        t0 = time.perf_counter()
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
        elapsed = time.perf_counter() - t0
        passed += ok
        cases.append((qid, bool(ok), detail, elapsed))
        print(f"{qid:28s} {detail:>6s}  {'PASS' if ok else 'FAIL'}")

    n = len(golden)
    acc = 100.0 * passed / n if n else 0.0
    print("-" * 48)
    print(f"Execution accuracy: {passed}/{n} ({acc:.1f}%)  [llm={llm.label}]")

    if args.no_history:
        print("history recording skipped (--no-history)")
    else:
        try:
            record_eval_history(datetime.date.today().isoformat(), passed, n,
                                acc, llm=llm.label)
        except Exception as e:
            # History is informational only: never fail an eval run over it.
            print(f"warning: could not record eval history: {e}")

    if args.junit:
        write_junit(args.junit, cases, llm.label)
        print(f"JUnit report written to {args.junit}")

    if args.json:
        verdict = verdict_dict(passed, n, args.fail_under)
        artifact = {
            **verdict,
            "passed": passed,
            "total": n,
            "llm": llm.label,
            "results": [
                {"id": qid, "ok": ok, "detail": detail,
                 "elapsed_s": round(elapsed, 3)}
                for qid, ok, detail, elapsed in cases
            ],
        }
        write_json(args.json, artifact)
        print(f"Verdict JSON written to {args.json}")

    return gate_exit(passed, n, args.fail_under)


def write_json(path: str, payload: dict) -> None:
    """Write a JSON artifact (stdlib only), creating parent dirs as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_junit(path: str, cases: list, llm_label: str) -> None:
    """Write per-question results as JUnit XML (stdlib only, CI-friendly)."""
    suite = ET.Element("testsuite", {
        "name": f"sql-rag-golden-evals[{llm_label}]",
        "tests": str(len(cases)),
        "failures": str(sum(1 for _, ok, _, _ in cases if not ok)),
        "errors": "0",
        "skipped": "0",
    })
    for qid, ok, detail, elapsed in cases:
        case = ET.SubElement(suite, "testcase", {
            "classname": "golden",
            "name": qid,
            "time": f"{elapsed:.3f}",
        })
        if not ok:
            failure = ET.SubElement(case, "failure", {"message": detail})
            failure.text = f"execution mismatch for {qid}: {detail}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tree = ET.ElementTree(suite)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    sys.exit(main())
