/* SQL RAG Studio — the workbench. One composer, one SQL block, one table.
   The chrome stays quiet; the SQL and the chart carry the color. */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type AskResponse,
  type Health,
  type HistoryMsg,
  type Row,
  type Schema,
} from "./api";
import Chart from "./components/Chart";
import EvalPanel from "./components/EvalPanel";
import SchemaPanel from "./components/SchemaPanel";
import SqlCode from "./components/SqlCode";

type View = "studio" | "evals";
type Tab = "results" | "sql" | "rag" | "history";

interface HistoryEntry {
  at: string;
  question: string;
  sql: string;
  rows: number;
  ok: boolean;
  genMs: number;
  execMs: number;
  repairs: number;
}

const HISTORY_KEY = "sql-rag-studio-history";

function loadHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function download(name: string, text: string, mime: string) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function rowsToCsv(columns: string[], rows: Row[]): string {
  const esc = (v: unknown) => {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [
    columns.map(esc).join(","),
    ...rows.map((r) => columns.map((c) => esc(r[c])).join(",")),
  ].join("\n");
}

function fmtCell(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    return Number.isInteger(v)
      ? v.toLocaleString("en-US")
      : v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  }
  return String(v);
}

export default function App() {
  const [view, setView] = useState<View>("studio");
  const [health, setHealth] = useState<Health | null>(null);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(0.2);
  const [wantSummary, setWantSummary] = useState(true);
  const [wantRepair, setWantRepair] = useState(true);
  const [wantFollowups, setWantFollowups] = useState(true);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [tab, setTab] = useState<Tab>("results");
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [chat, setChat] = useState<HistoryMsg[]>([]);
  const [editedSql, setEditedSql] = useState("");
  const [plan, setPlan] = useState("");
  const [planLoading, setPlanLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.schema().then(setSchema).catch(() => setSchema(null));
    api.examples().then(setExamples).catch(() => setExamples([]));
  }, []);

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 50)));
  }, [history]);

  const ask = useCallback(
    async (q: string) => {
      const query = q.trim();
      if (!query || asking) return;
      setAsking(true);
      setError("");
      try {
        const res = await api.ask({
          question: query,
          top_k: topK,
          temperature,
          auto_repair: wantRepair,
          summarize: wantSummary,
          history: wantFollowups ? chat : [],
        });
        setAnswer(res);
        setEditedSql(res.sql);
        setPlan("");
        setTab("results");
        setHistory((h) => [
          {
            at: new Date().toLocaleTimeString("en-GB"),
            question: query,
            sql: res.sql,
            rows: res.row_count,
            ok: res.ok,
            genMs: res.gen_time_ms,
            execMs: res.exec_time_ms,
            repairs: res.repair_attempts.length,
          },
          ...h,
        ]);
        if (wantFollowups) {
          setChat((c) => [
            ...c.slice(-8),
            { role: "user", content: query },
            { role: "assistant", content: `Ran SQL: ${res.sql}` },
          ]);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setAsking(false);
      }
    },
    [asking, chat, temperature, topK, wantFollowups, wantRepair, wantSummary],
  );

  const runEdited = async () => {
    if (!editedSql.trim()) return;
    setError("");
    try {
      const res = await api.execute(editedSql);
      setAnswer((a) =>
        a
          ? {
              ...a,
              sql: res.sql,
              status: res.status,
              ok: res.ok,
              columns: res.columns,
              rows: res.rows,
              row_count: res.row_count,
              chart: res.chart,
              exec_time_ms: res.exec_time_ms,
            }
          : a,
      );
      setPlan("");
      setTab("results");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const loadPlan = async () => {
    const sql = (answer?.sql || "").trim();
    if (!sql) return;
    setPlanLoading(true);
    try {
      const res = await api.explain(sql);
      setPlan(res.plan);
    } catch (e) {
      setPlan(`EXPLAIN failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setPlanLoading(false);
    }
  };

  const copySql = async () => {
    if (!answer?.sql) return;
    try {
      await navigator.clipboard.writeText(answer.sql);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  const llmOk = health?.llm_available ?? false;
  const lampClass = !health ? "lamp-off" : llmOk ? "lamp-ok" : "lamp-warn";
  const lampText = !health
    ? "Connecting…"
    : llmOk
      ? `Ready · ${health.llm_provider}:${health.llm_model}`
      : `LLM unavailable · ${health.llm_detail}`;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">▦</span>
          <span className="brand-name">SQL RAG Studio</span>
        </div>
        <nav className="view-switch" aria-label="Views">
          <button
            className={view === "studio" ? "view-btn active" : "view-btn"}
            onClick={() => setView("studio")}
          >
            Studio
          </button>
          <button
            className={view === "evals" ? "view-btn active" : "view-btn"}
            onClick={() => setView("evals")}
          >
            Evals
          </button>
        </nav>
        <div className="status" title={health?.llm_detail || ""}>
          <span className={`lamp ${lampClass}`} />
          <span className="status-text mono">{lampText}</span>
        </div>
      </header>

      <div className="shell">
        {view === "studio" && <SchemaPanel schema={schema} />}

        <main className={view === "studio" ? "main" : "main main-wide"}>
          {view === "evals" ? (
            <EvalPanel llmAvailable={llmOk} />
          ) : (
            <>
              <section className="card composer">
                <label className="composer-label" htmlFor="question">
                  Ask in plain English
                </label>
                <textarea
                  id="question"
                  ref={composerRef}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(question);
                  }}
                  placeholder="Show total sales revenue by product category, ordered by revenue descending"
                  rows={3}
                />
                <div className="composer-row">
                  <div className="examples">
                    {examples.map((ex) => (
                      <button
                        key={ex}
                        className="chip"
                        onClick={() => {
                          setQuestion(ex);
                          composerRef.current?.focus();
                        }}
                      >
                        {ex}
                      </button>
                    ))}
                  </div>
                  <button
                    className="btn btn-primary ask-btn"
                    onClick={() => ask(question)}
                    disabled={asking || !question.trim() || !llmOk}
                    title={!llmOk ? "LLM unavailable" : "Ask (Ctrl+Enter)"}
                  >
                    {asking ? "Generating…" : "Generate and run"}
                  </button>
                </div>
                <div className="options-row">
                  <label className="opt">
                    <input type="checkbox" checked={wantSummary}
                      onChange={(e) => setWantSummary(e.target.checked)} />
                    Summary
                  </label>
                  <label className="opt">
                    <input type="checkbox" checked={wantRepair}
                      onChange={(e) => setWantRepair(e.target.checked)} />
                    Auto-repair
                  </label>
                  <label className="opt">
                    <input type="checkbox" checked={wantFollowups}
                      onChange={(e) => setWantFollowups(e.target.checked)} />
                    Follow-ups
                  </label>
                  <label className="opt opt-select">
                    Top K
                    <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                      {[3, 5, 8, 10].map((k) => (
                        <option key={k} value={k}>{k}</option>
                      ))}
                    </select>
                  </label>
                  <label className="opt opt-select">
                    Temperature
                    <select value={temperature}
                      onChange={(e) => setTemperature(Number(e.target.value))}>
                      <option value={0}>0.0</option>
                      <option value={0.2}>0.2</option>
                      <option value={0.5}>0.5</option>
                    </select>
                  </label>
                  {chat.length > 0 && (
                    <button className="link-btn" onClick={() => setChat([])}>
                      Clear conversation
                    </button>
                  )}
                </div>
              </section>

              {error && <div className="alert alert-error">{error}</div>}

              {!answer && !error && (
                <section className="card empty-state">
                  <h2>Ask the database a question</h2>
                  <p className="muted">
                    The studio retrieves the relevant tables, prunes the schema
                    to what matters, and asks the model for one read-only query
                    — then shows you the SQL, the plan, and the rows. Try an
                    example above, or ask your own.
                  </p>
                  <div className="empty-meta mono">
                    {schema
                      ? `${schema.database} · ${schema.tables
                          .map((t) => t.name)
                          .join(" · ")}`
                      : "Loading database…"}
                  </div>
                </section>
              )}

              {answer && (
                <>
                  <section className="card readout">
                    <div className="readout-line">
                      <span className={`lamp ${answer.ok ? "lamp-ok" : "lamp-bad"}`} />
                      <span className="readout-status">{answer.status}</span>
                    </div>
                    <div className="readout-meta mono">
                      <span>{answer.row_count} rows</span>
                      <span>gen {answer.gen_time_ms.toFixed(0)} ms</span>
                      <span>exec {answer.exec_time_ms.toFixed(1)} ms</span>
                      <span>tables: {answer.tables_used.join(", ") || "—"}</span>
                      {answer.pruned && <span>schema pruned</span>}
                      <span>{answer.llm}</span>
                    </div>
                    {answer.rewritten_question &&
                      answer.rewritten_question !== answer.question && (
                        <div className="rewritten">
                          Follow-up interpreted as: “{answer.rewritten_question}”
                        </div>
                      )}
                    {answer.summary && (
                      <p className="summary">{answer.summary}</p>
                    )}
                  </section>

                  <nav className="tabs" aria-label="Result views">
                    {(["results", "sql", "rag", "history"] as Tab[]).map((t) => (
                      <button
                        key={t}
                        className={tab === t ? "tab active" : "tab"}
                        onClick={() => setTab(t)}
                      >
                        {t === "results"
                          ? "Results"
                          : t === "sql"
                            ? "SQL and plan"
                            : t === "rag"
                              ? "RAG diagnostics"
                              : `History (${history.length})`}
                      </button>
                    ))}
                  </nav>

                  {tab === "results" && (
                    <section className="card">
                      <div className="metrics">
                        <div className="metric">
                          <div className="metric-label">Rows</div>
                          <div className="metric-value">{answer.row_count.toLocaleString("en-US")}</div>
                        </div>
                        <div className="metric">
                          <div className="metric-label">Columns</div>
                          <div className="metric-value">{answer.columns.length}</div>
                        </div>
                        <div className="metric">
                          <div className="metric-label">Numeric columns</div>
                          <div className="metric-value">
                            {answer.columns.filter((c) =>
                              answer.rows.some((r) => typeof r[c] === "number"),
                            ).length}
                          </div>
                        </div>
                        <div className="metric-actions">
                          <button
                            className="btn btn-secondary btn-sm"
                            disabled={answer.rows.length === 0}
                            onClick={() =>
                              download("sql_results.csv", rowsToCsv(answer.columns, answer.rows), "text/csv")
                            }
                          >
                            Download CSV
                          </button>
                          <button
                            className="btn btn-secondary btn-sm"
                            disabled={answer.rows.length === 0}
                            onClick={() =>
                              download("sql_results.json", JSON.stringify(answer.rows, null, 2), "application/json")
                            }
                          >
                            Download JSON
                          </button>
                        </div>
                      </div>

                      {answer.chart.type !== "table" && answer.rows.length > 0 && (
                        <div className="chart-block">
                          <div className="chart-title mono">
                            {answer.chart.type} · {answer.chart.x}
                            {answer.chart.y ? ` × ${answer.chart.y}` : ""}
                          </div>
                          <Chart chart={answer.chart} rows={answer.rows} />
                        </div>
                      )}

                      {answer.rows.length > 0 ? (
                        <div className="table-wrap">
                          <table className="data-table">
                            <thead>
                              <tr>
                                {answer.columns.map((c) => (
                                  <th key={c}>{c}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {answer.rows.map((r, i) => (
                                <tr key={i}>
                                  {answer.columns.map((c) => (
                                    <td
                                      key={c}
                                      className={typeof r[c] === "number" ? "num" : ""}
                                    >
                                      {fmtCell(r[c])}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <p className="muted">No rows returned by this query.</p>
                      )}
                    </section>
                  )}

                  {tab === "sql" && (
                    <section className="card">
                      <div className="section-head">
                        <h3>Generated SQL</h3>
                        <button className="btn btn-secondary btn-sm" onClick={copySql}>
                          {copied ? "Copied" : "Copy SQL"}
                        </button>
                      </div>
                      <SqlCode sql={answer.sql} />

                      <div className="section-head plan-head">
                        <h3>Query plan</h3>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={loadPlan}
                          disabled={planLoading}
                        >
                          {planLoading ? "Loading…" : "Show EXPLAIN QUERY PLAN"}
                        </button>
                      </div>
                      {plan && (
                        <pre className="plan-code"><code>{plan}</code></pre>
                      )}

                      <div className="section-head">
                        <h3>Edit and re-run</h3>
                      </div>
                      <p className="muted">
                        Modify the SQL and run it directly. The same safety
                        rules apply: read-only, validated, capped.
                      </p>
                      <textarea
                        className="sql-editor"
                        value={editedSql}
                        onChange={(e) => setEditedSql(e.target.value)}
                        rows={6}
                        spellCheck={false}
                      />
                      <button className="btn btn-secondary" onClick={runEdited}
                        disabled={!editedSql.trim()}>
                        Execute edited SQL
                      </button>
                    </section>
                  )}

                  {tab === "rag" && (
                    <section className="card">
                      <h3>Retrieved schema chunks</h3>
                      {answer.chunks.length > 0 ? (
                        <ul className="chunk-list">
                          {answer.chunks.map((c, i) => (
                            <li key={i} className="chunk">
                              <div className="chunk-head mono">
                                <span>{c.type}</span>
                                <span>relevance {c.score.toFixed(3)}</span>
                              </div>
                              <div className="chunk-text">{c.text}</div>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="muted">No vector chunks retrieved.</p>
                      )}

                      <h3 className="section-gap">Tables sent to the model</h3>
                      <div className="tag-row">
                        {answer.tables_used.map((t) => (
                          <span key={t} className="tag mono">{t}</span>
                        ))}
                        {answer.pruned && (
                          <span className="tag tag-accent mono">pruned</span>
                        )}
                      </div>

                      {answer.repair_attempts.length > 0 && (
                        <>
                          <h3 className="section-gap">Auto-repair attempts</h3>
                          <ul className="chunk-list">
                            {answer.repair_attempts.map((a) => (
                              <li key={a.attempt} className="chunk">
                                <div className="chunk-head mono">
                                  <span>Attempt {a.attempt}</span>
                                  <span className={a.error ? "text-bad" : "text-ok"}>
                                    {a.error ? "failed" : "fixed"}
                                  </span>
                                </div>
                                <SqlCode sql={a.sql} />
                                {a.error && <div className="muted">{a.error}</div>}
                              </li>
                            ))}
                          </ul>
                        </>
                      )}
                    </section>
                  )}

                  {tab === "history" && (
                    <section className="card">
                      <h3>Execution history</h3>
                      {history.length > 0 ? (
                        <ul className="history-list">
                          {history.map((h, i) => (
                            <li key={i} className="history-item">
                              <div className="history-q">
                                <span className="mono muted">{h.at}</span>{" "}
                                {h.question}
                              </div>
                              <SqlCode sql={h.sql} />
                              <div className="history-meta mono muted">
                                {h.rows} rows · {h.ok ? "ok" : "error"} · gen{" "}
                                {h.genMs.toFixed(0)} ms · exec {h.execMs.toFixed(1)} ms
                                {h.repairs > 0 ? ` · ${h.repairs} repairs` : ""}
                              </div>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="muted">No queries run yet.</p>
                      )}
                    </section>
                  )}
                </>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
