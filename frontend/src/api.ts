/* Typed client for the FastAPI backend (same origin in production,
   proxied to :8000 by Vite in dev). */

export interface Health {
  status: string;
  database: string;
  table_count: number;
  tables: string[];
  embedding_backend: string;
  llm_provider: string;
  llm_model: string;
  llm_available: boolean;
  llm_detail: string;
}

export interface TableColumn {
  name: string;
  type: string;
  is_pk?: boolean;
}

export interface ForeignKey {
  from_column: string;
  to_table: string;
  to_column: string;
}

export interface TableInfo {
  name: string;
  row_count: number;
  columns: TableColumn[];
  foreign_keys: ForeignKey[];
}

export interface Schema {
  database: string;
  db_path: string;
  table_count: number;
  tables: TableInfo[];
}

export interface Chunk {
  text: string;
  type: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface RepairAttempt {
  attempt: number;
  sql: string;
  error: string | null;
}

export interface ChartSpec {
  type: string;
  x: string | null;
  y: string | null;
}

export type Row = Record<string, string | number | boolean | null>;

export interface AskResponse {
  question: string;
  rewritten_question: string;
  sql: string;
  status: string;
  ok: boolean;
  columns: string[];
  rows: Row[];
  row_count: number;
  summary: string;
  chart: ChartSpec;
  chunks: Chunk[];
  tables_used: string[];
  pruned: boolean;
  repair_attempts: RepairAttempt[];
  gen_time_ms: number;
  exec_time_ms: number;
  llm: string;
}

export interface ExecuteResponse {
  sql: string;
  status: string;
  ok: boolean;
  columns: string[];
  rows: Row[];
  row_count: number;
  chart: ChartSpec;
  exec_time_ms: number;
}

export interface EvalQuestion {
  id: string;
  question: string;
}

export interface EvalResult {
  id: string;
  question: string;
  ok: boolean;
  detail: string;
  sql: string;
  elapsed_s: number;
}

export interface EvalRun {
  run_id: string;
  status: "running" | "done" | "error";
  mock: boolean;
  passed: number;
  total: number;
  accuracy: number;
  llm: string;
  results: EvalResult[];
  error?: string;
}

export interface HistoryMsg {
  role: string;
  content: string;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      (body as { detail?: string }).detail || `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return body as T;
}

export const api = {
  health: () => req<Health>("/api/health"),
  schema: () => req<Schema>("/api/schema"),
  examples: () =>
    req<{ examples: string[] }>("/api/examples").then((r) => r.examples),
  ask: (payload: {
    question: string;
    top_k: number;
    temperature: number;
    auto_repair: boolean;
    summarize: boolean;
    history: HistoryMsg[];
  }) =>
    req<AskResponse>("/api/ask", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  execute: (sql: string) =>
    req<ExecuteResponse>("/api/execute", {
      method: "POST",
      body: JSON.stringify({ sql }),
    }),
  explain: (sql: string) =>
    req<{ sql: string; plan: string }>("/api/explain", {
      method: "POST",
      body: JSON.stringify({ sql }),
    }),
  evalQuestions: () =>
    req<{ total: number; questions: EvalQuestion[] }>("/api/eval/questions"),
  startEval: (mock: boolean) =>
    req<{ run_id: string; status: string; mock: boolean }>("/api/eval", {
      method: "POST",
      body: JSON.stringify({ mock }),
    }),
  evalStatus: (runId: string) => req<EvalRun>(`/api/eval/${runId}`),
};
