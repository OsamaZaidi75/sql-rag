"""FastAPI backend for SQL RAG Studio.

Serves the React frontend (built files in ``frontend/dist``) and a JSON API
over the existing ``core/`` engine. The LLM provider is configured purely
from environment variables — see :mod:`api.config`.
"""

from __future__ import annotations
