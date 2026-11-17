# SQL RAG Studio — single container: FastAPI backend + built React frontend.
# Multi-arch safe: node:22-alpine and python:3.12-slim both ship ARM64 images,
# and all Python deps have aarch64 wheels (no compilation step).
#
# Run:
#   docker build -t sql-rag .
#   docker run --env-file .env -p 8000:8000 sql-rag
# The only required env var is NVIDIA_API_KEY (see README "Configuration").
# The sample DB self-initializes on first start; no volume is required.
# Mount /app/data to persist the DB and the vector-index cache across restarts.

# --- Stage 1: build the React frontend ---------------------------------
FROM node:22-alpine AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python runtime -------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# System deps: curl for the healthcheck only.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY api/ ./api/
COPY evals/ ./evals/
COPY app.py README.md ./
COPY data/.gitkeep ./data/.gitkeep
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

# Runtime user owns /app/data (sample DB + vector-index cache are written
# there on first start). Nothing else in the image is written at runtime.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/reports/daily \
    && chown -R app:app /app/data /app/reports
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

CMD ["python", "-m", "api.main"]
