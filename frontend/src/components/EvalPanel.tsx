/* Golden-set eval runner: lists the questions, runs the set against the
   mock harness (no LLM) or the live model, and polls for results. */

import { useEffect, useRef, useState } from "react";
import { api, type EvalQuestion, type EvalRun } from "../api";

export default function EvalPanel({ llmAvailable }: { llmAvailable: boolean }) {
  const [questions, setQuestions] = useState<EvalQuestion[]>([]);
  const [total, setTotal] = useState(0);
  const [run, setRun] = useState<EvalRun | null>(null);
  const [error, setError] = useState("");
  const timer = useRef<number | null>(null);

  useEffect(() => {
    api
      .evalQuestions()
      .then((r) => {
        setQuestions(r.questions);
        setTotal(r.total);
      })
      .catch((e) => setError(e.message));
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, []);

  const poll = (runId: string) => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(() => {
      api
        .evalStatus(runId)
        .then((r) => {
          setRun(r);
          if (r.status !== "running" && timer.current) {
            window.clearInterval(timer.current);
          }
        })
        .catch((e) => {
          setError(e.message);
          if (timer.current) window.clearInterval(timer.current);
        });
    }, 1500);
  };

  const start = async (mock: boolean) => {
    setError("");
    setRun(null);
    try {
      const r = await api.startEval(mock);
      setRun({
        run_id: r.run_id,
        status: "running",
        mock,
        passed: 0,
        total,
        accuracy: 0,
        llm: mock ? "mock" : "",
        results: [],
      });
      poll(r.run_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const done = run?.results.length ?? 0;
  const progress = run && run.total > 0 ? (done / run.total) * 100 : 0;

  return (
    <div className="eval-view">
      <div className="eval-head card">
        <div>
          <h2>Golden evals</h2>
          <p className="muted">
            {total} questions scored by execution match: the generated SQL
            passes when it returns the same rows as the golden answer,
            order-insensitive. SQL strings are never compared.
          </p>
        </div>
        <div className="eval-actions">
          <button className="btn btn-secondary" onClick={() => start(true)}
            disabled={run?.status === "running"}>
            Run mock harness
          </button>
          <button className="btn btn-primary" onClick={() => start(false)}
            disabled={run?.status === "running" || !llmAvailable}
            title={llmAvailable ? "" : "LLM unavailable"}>
            Run with live model
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {run && (
        <div className="card eval-progress">
          <div className="eval-meta">
            <span className="mono">
              {run.status === "running"
                ? `Running… ${done}/${run.total}`
                : `${run.passed}/${run.total} passed · ${run.accuracy}% accuracy`}
            </span>
            <span className="muted mono">llm: {run.llm || "—"}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          {run.status === "error" && (
            <div className="alert alert-error">{run.error}</div>
          )}
          {run.results.length > 0 && (
            <table className="data-table eval-table">
              <thead>
                <tr>
                  <th>Question</th>
                  <th>Result</th>
                  <th className="num">Time</th>
                </tr>
              </thead>
              <tbody>
                {run.results.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <div className="eval-q">{r.question}</div>
                      <div className="muted mono eval-id">{r.id}</div>
                    </td>
                    <td>
                      <span className={r.ok ? "pill pill-ok" : "pill pill-bad"}>
                        {r.ok ? "Pass" : "Fail"}
                      </span>{" "}
                      <span className="muted">{r.detail}</span>
                    </td>
                    <td className="num mono">{r.elapsed_s.toFixed(2)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      <div className="card">
        <h3>The golden set</h3>
        <ol className="golden-list">
          {questions.map((q) => (
            <li key={q.id}>
              <span className="eval-q">{q.question}</span>
              <span className="muted mono eval-id">{q.id}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
