"""FastAPI entrypoint for SQL RAG Studio.

JSON API under ``/api/*`` plus the built React frontend (``frontend/dist``)
served from the same process, so one container is the whole deployment.

Run locally:

    uvicorn api.main:app --reload --port 8000

Or: ``python -m api.main`` (reads ``HOST``/``PORT`` from the environment).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_settings
from .engine import DEFAULT_TEMPERATURE, Engine, engine, load_golden

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST = os.path.join(ROOT, "frontend", "dist")

app = FastAPI(
    title="SQL RAG Studio",
    description="Natural language to SQL over SQLite, with hybrid RAG retrieval.",
    version="2.0.0",
)


def get_engine() -> Engine:
    """Indirection so tests can swap the process engine."""
    return engine


# -- request models -----------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=10)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=1.0)
    auto_repair: bool = True
    summarize: bool = True
    history: Optional[List[Dict[str, str]]] = None


class SqlRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=10000)


class EvalRequest(BaseModel):
    mock: bool = False


# -- health / config ----------------------------------------------------


@app.get("/api/health")
def health() -> Dict[str, Any]:
    eng = get_engine()
    eng.ensure_db()
    schema = eng.schema()
    llm = eng.llm_status()
    return {
        "status": "ok",
        "database": schema["database"],
        "table_count": schema["table_count"],
        "tables": [t["name"] for t in schema["tables"]],
        "embedding_backend": eng.settings.embedding_backend,
        "llm_provider": eng.settings.llm_provider,
        "llm_model": eng.settings.llm_model,
        "llm_available": llm["available"],
        "llm_detail": llm["detail"],
    }


@app.get("/api/config")
def config() -> Dict[str, Any]:
    return get_engine().settings.public_dict()


# -- schema -------------------------------------------------------------


@app.get("/api/schema")
def schema() -> Dict[str, Any]:
    return get_engine().schema()


@app.get("/api/schema/{table}")
def table_schema(table: str) -> Dict[str, Any]:
    details = get_engine().table_details(table)
    if not details:
        raise HTTPException(status_code=404, detail=f"no such table: {table}")
    return {"name": table, **details}


# -- ask / execute / explain --------------------------------------------


@app.post("/api/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be blank")
    eng = get_engine()
    status = eng.llm_status()
    if not status["available"]:
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable: {status['detail']}",
        )
    try:
        return eng.ask(
            question=req.question.strip(),
            top_k=req.top_k,
            temperature=req.temperature,
            auto_repair=req.auto_repair,
            summarize=req.summarize,
            history=req.history,
        )
    except Exception as e:  # never leak a stack trace to the UI
        raise HTTPException(status_code=500, detail=f"ask failed: {e}")


@app.post("/api/execute")
def execute(req: SqlRequest) -> Dict[str, Any]:
    return get_engine().execute_sql(req.sql)


@app.post("/api/explain")
def explain(req: SqlRequest) -> Dict[str, Any]:
    return get_engine().explain(req.sql)


# -- examples / evals ---------------------------------------------------


@app.get("/api/examples")
def examples() -> Dict[str, Any]:
    """Four starter questions, drawn from the golden set."""
    golden = load_golden()
    picks = ["top3_salaries", "revenue_per_region", "employees_per_department",
             "avg_rating_by_city"]
    by_id = {item["id"]: item["question"] for item in golden}
    chosen = [by_id[p] for p in picks if p in by_id]
    if len(chosen) < 4:  # fall back to the first questions if ids change
        chosen = [item["question"] for item in golden[:4]]
    return {"examples": chosen}


@app.get("/api/eval/questions")
def eval_questions() -> Dict[str, Any]:
    golden = load_golden()
    return {
        "total": len(golden),
        "questions": [
            {"id": item["id"], "question": item["question"]} for item in golden
        ],
    }


@app.post("/api/eval", status_code=202)
def start_eval(req: EvalRequest) -> Dict[str, Any]:
    eng = get_engine()
    if not req.mock:
        status = eng.llm_status()
        if not status["available"]:
            raise HTTPException(
                status_code=503,
                detail=f"LLM unavailable: {status['detail']}",
            )
    return eng.start_eval(mock=req.mock)


@app.get("/api/eval/{run_id}")
def eval_status(run_id: str) -> Dict[str, Any]:
    run = get_engine().get_eval(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no such eval run: {run_id}")
    return run


# -- frontend -----------------------------------------------------------

if os.path.isdir(FRONTEND_DIST):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")),
        name="assets",
    )

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        """Serve the React app; unknown non-API paths fall back to index."""
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="no such API route")
        candidate = os.path.join(FRONTEND_DIST, path)
        if path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

else:

    @app.get("/", include_in_schema=False)
    def index() -> JSONResponse:
        return JSONResponse(
            {
                "service": "sql-rag-studio",
                "frontend": "not built (run `npm run build` in frontend/)",
                "api_docs": "/docs",
                "health": "/api/health",
            }
        )


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
