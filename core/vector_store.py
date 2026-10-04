"""
Hybrid Vector Store & Schema Indexer for SQL RAG
Supports multiple embedding backends (Ollama, SentenceTransformers, TF-IDF fallback)
with automatic failover, batch processing, and normalized cosine similarity search.

Retrieval is genuinely hybrid: dense cosine similarity and BM25 keyword search
run independently and their ranked lists are fused with Reciprocal Rank Fusion
(RRF), so exact table/column name matches and semantic matches reinforce
each other instead of one backend merely replacing the other.

The built index can be persisted to disk and reloaded when the underlying
database file has not changed (checked via mtime), so restarts don't pay
the embedding cost again.
"""

import json
import os
import pickle
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer shared by BM25 indexing and querying."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(ranked_id_lists: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion over ranked id lists.
    score(id) = sum(1 / (k + rank)) across lists. Returns (id, score) sorted desc.
    """
    scores: Dict[int, float] = {}
    for ids in ranked_id_lists:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


class VectorStore:
    def __init__(self, backend: str = "auto", ollama_model: str = "nomic-embed-text", hf_model: str = "BAAI/bge-small-en-v1.5"):
        self.backend = backend  # 'ollama', 'sentence-transformers', 'tfidf', 'auto'
        self.ollama_model = ollama_model
        self.hf_model_name = hf_model
        self.active_backend = "tfidf"
        self._hf_model = None
        self._tfidf_vectorizer = None
        self._bm25 = None

        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Discovers and initializes the best available embedding engine."""
        if self.backend in ("ollama", "auto"):
            try:
                import ollama
                test_resp = ollama.embed(model=self.ollama_model, input="schema test")
                embeddings = getattr(test_resp, "embeddings", None) or test_resp.get("embeddings")
                if embeddings and len(embeddings) > 0:
                    self.active_backend = "ollama"
                    return
            except Exception:
                if self.backend == "ollama":
                    pass

        if self.backend in ("sentence-transformers", "auto"):
            try:
                from sentence_transformers import SentenceTransformer
                self._hf_model = SentenceTransformer(self.hf_model_name)
                self.active_backend = "sentence-transformers"
                return
            except Exception:
                if self.backend == "sentence-transformers":
                    pass

        # Fallback to TF-IDF (100% offline, zero network, zero crash)
        self.active_backend = "tfidf"

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts using the active backend."""
        if not texts:
            return np.empty((0, 0))

        if self.active_backend == "ollama":
            try:
                import ollama
                resp = ollama.embed(model=self.ollama_model, input=texts)
                raw_emb = getattr(resp, "embeddings", None) or resp.get("embeddings")
                arr = np.array(raw_emb, dtype=np.float32)
                # Normalize for cosine similarity
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
                return arr / norms
            except Exception:
                # Fallback to TF-IDF
                self.active_backend = "tfidf"

        if self.active_backend == "sentence-transformers":
            try:
                if self._hf_model is None:
                    from sentence_transformers import SentenceTransformer
                    self._hf_model = SentenceTransformer(self.hf_model_name)
                arr = self._hf_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return np.array(arr, dtype=np.float32)
            except Exception:
                self.active_backend = "tfidf"

        # TF-IDF Fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        if self._tfidf_vectorizer is None or len(self.chunks) == 0:
            self._tfidf_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True
            )
            # Fit on the current texts
            dense = self._tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            dense = self._tfidf_vectorizer.transform(texts).toarray()

        arr = np.array(dense, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr / norms

    def _embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string with the active backend."""
        if self.active_backend == "tfidf" and self._tfidf_vectorizer is not None:
            q_vec = self._tfidf_vectorizer.transform([query]).toarray()
            q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9
            return (q_vec / q_norm).astype(np.float32)
        return self._embed_texts([query])

    def add_chunk(self, text: str, metadata: Dict[str, Any], doc_id: str = ""):
        """Adds a document chunk to the vector store buffer."""
        chunk = DocumentChunk(text=text.strip(), metadata=metadata, doc_id=doc_id)
        self.chunks.append(chunk)

    def build_index(self):
        """Computes embeddings for all accumulated chunks, plus the BM25 index."""
        if not self.chunks:
            self.embeddings = None
            self._bm25 = None
            return

        texts = [chunk.text for chunk in self.chunks]

        # If TF-IDF, reset vectorizer so it fits all chunk texts
        if self.active_backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True
            )
            matrix = self._tfidf_vectorizer.fit_transform(texts).toarray()
            arr = np.array(matrix, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            self.embeddings = arr / norms
        else:
            self.embeddings = self._embed_texts(texts)

        # BM25 keyword index over the same chunks (graceful if dep missing)
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([_tokenize(t) for t in texts])
        except Exception:
            self._bm25 = None

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> List[Tuple[DocumentChunk, float]]:
        """
        Dense/cosine search over the index. Returns list of (DocumentChunk, similarity_score).
        For fused dense+BM25 retrieval, use search_hybrid().
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_emb = self._embed_query(query)
        if q_emb.shape[0] == 0:
            return []

        # Cosine similarity (vectors are L2-normalized)
        sims = np.dot(self.embeddings, q_emb[0]).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score >= min_score:
                results.append((self.chunks[idx], score))

        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 25,
        bm25_k: int = 25,
        rrf_k: int = 60,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Hybrid retrieval: dense cosine ranking and BM25 keyword ranking are
        fused with Reciprocal Rank Fusion. Returns (chunk, rrf_score) sorted
        by fused score. Falls back to dense-only when BM25 is unavailable.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q_emb = self._embed_query(query)
        if q_emb.shape[0] == 0:
            return []

        sims = np.dot(self.embeddings, q_emb[0]).flatten()
        dense_ids = np.argsort(sims)[::-1][:dense_k].tolist()

        kw_ids: List[int] = []
        if self._bm25 is not None:
            try:
                bm25_scores = self._bm25.get_scores(_tokenize(query))
                ranked = np.argsort(bm25_scores)[::-1][:bm25_k]
                kw_ids = [int(i) for i in ranked if bm25_scores[i] > 0]
            except Exception:
                kw_ids = []

        ranked_lists = [dense_ids, kw_ids] if kw_ids else [dense_ids]
        fused = rrf_fuse(ranked_lists, k=rrf_k)[:top_k]
        return [(self.chunks[i], float(score)) for i, score in fused]

    # -- persistence ----------------------------------------------------

    def save_index(self, base_path: str, db_mtime: Optional[float] = None):
        """
        Persists the built index to disk. Files written:
        ``<base>.npz`` (embeddings), ``<base>.chunks.json`` (chunk data),
        ``<base>.json`` (manifest), ``<base>.tfidf.pkl`` (TF-IDF vectorizer, if used).
        """
        if self.embeddings is None:
            raise RuntimeError("Cannot save an index that has not been built.")

        base = str(base_path)
        np.savez_compressed(base + ".npz", embeddings=self.embeddings)
        with open(base + ".chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f)
        manifest = {
            "active_backend": self.active_backend,
            "embedding_dim": int(self.embeddings.shape[1]),
            "chunk_count": len(self.chunks),
            "db_mtime": db_mtime,
        }
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        if self.active_backend == "tfidf" and self._tfidf_vectorizer is not None:
            with open(base + ".tfidf.pkl", "wb") as f:
                pickle.dump(self._tfidf_vectorizer, f)

    @classmethod
    def load_index(cls, base_path: str, **kwargs) -> "VectorStore":
        """Reloads an index previously written by :meth:`save_index`."""
        base = str(base_path)
        with open(base + ".json", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(base + ".chunks.json", encoding="utf-8") as f:
            chunk_dicts = json.load(f)

        store = cls.__new__(cls)
        store.backend = kwargs.get("backend", "auto")
        store.ollama_model = kwargs.get("ollama_model", "nomic-embed-text")
        store.hf_model_name = kwargs.get("hf_model", "BAAI/bge-small-en-v1.5")
        store.active_backend = manifest.get("active_backend", "tfidf")
        store._hf_model = None
        store._tfidf_vectorizer = None
        store._bm25 = None
        store.chunks = [DocumentChunk(**c) for c in chunk_dicts]
        store.embeddings = np.load(base + ".npz")["embeddings"]

        if store.active_backend == "tfidf":
            tfidf_path = base + ".tfidf.pkl"
            if os.path.exists(tfidf_path):
                with open(tfidf_path, "rb") as f:
                    store._tfidf_vectorizer = pickle.load(f)
        try:
            from rank_bm25 import BM25Okapi
            store._bm25 = BM25Okapi([_tokenize(c.text) for c in store.chunks])
        except Exception:
            store._bm25 = None
        return store


def index_cache_path(db_path: str, backend: str, cache_dir: str = "data/index_cache") -> str:
    """Deterministic cache base path for a database + embedding backend."""
    stem = Path(db_path).stem
    safe_backend = re.sub(r"[^a-z0-9_-]", "_", backend.lower())
    return os.path.join(cache_dir, f"{stem}.{safe_backend}")


def get_or_build_index(
    db_path: str,
    backend: str = "auto",
    ollama_embed_model: str = "nomic-embed-text",
    cache_dir: str = "data/index_cache",
) -> VectorStore:
    """
    Returns a built VectorStore for ``db_path``, loading a cached index from
    disk when the database file is unchanged (mtime check), otherwise building
    fresh and caching it.
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = index_cache_path(db_path, backend, cache_dir)
    manifest_path = base + ".json"
    db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else None

    if os.path.exists(manifest_path) and db_mtime is not None:
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("db_mtime") == db_mtime and manifest.get("chunk_count"):
                return VectorStore.load_index(
                    base, backend=backend, ollama_model=ollama_embed_model
                )
        except Exception:
            pass  # corrupt/partial cache -> rebuild

    store = VectorStore(backend=backend, ollama_model=ollama_embed_model)
    index_database_schema(db_path, store)
    try:
        store.save_index(base, db_mtime=db_mtime)
    except Exception:
        pass  # caching is best-effort; a live index is what matters
    return store


def index_database_schema(db_path: str, vector_store: VectorStore):
    """
    Extracts granular schema features from SQLite database and indexes them into vector store.
    """
    from .database import get_full_database_schema

    schema_map = get_full_database_schema(db_path)
    vector_store.chunks.clear()

    for table_name, details in schema_map.items():
        # 1. Table overview chunk
        col_names = [c["name"] for c in details["columns"]]
        table_text = (
            f"Table `{table_name}` contains {details['row_count']} records. "
            f"Columns: {', '.join(col_names)}."
        )
        vector_store.add_chunk(
            text=table_text,
            metadata={"type": "table_summary", "table": table_name},
            doc_id=f"table_{table_name}"
        )

        # 2. Individual Column Chunks with data types and primary key markers
        for col in details["columns"]:
            pk_note = " (PRIMARY KEY)" if col["is_pk"] else ""
            col_text = f"Table `{table_name}` has column `{col['name']}` with data type {col['type']}{pk_note}."
            vector_store.add_chunk(
                text=col_text,
                metadata={"type": "column", "table": table_name, "column": col["name"], "dtype": col["type"]},
                doc_id=f"col_{table_name}_{col['name']}"
            )

        # 3. Categorical Sample Values for string columns
        for col_name, sample_vals in details.get("sample_values", {}).items():
            if sample_vals and any(isinstance(v, str) for v in sample_vals):
                val_str = ", ".join([f"'{v}'" for v in sample_vals[:6]])
                sample_text = (
                    f"Table `{table_name}` column `{col_name}` has distinct sample values: {val_str}. "
                    f"Use these exact string literals when filtering `{table_name}.{col_name}` in WHERE clauses."
                )
                vector_store.add_chunk(
                    text=sample_text,
                    metadata={"type": "sample_values", "table": table_name, "column": col_name},
                    doc_id=f"samples_{table_name}_{col_name}"
                )

        # 4. Foreign Key Relationships
        for fk in details.get("foreign_keys", []):
            fk_text = (
                f"Relationship: Table `{table_name}` column `{fk['from_column']}` joins with "
                f"`{fk['to_table']}` column `{fk['to_column']}` (JOIN `{table_name}` ON `{table_name}`.{fk['from_column']} = `{fk['to_table']}.{fk['to_column']}`)."
            )
            vector_store.add_chunk(
                text=fk_text,
                metadata={"type": "foreign_key", "from_table": table_name, "to_table": fk["to_table"]},
                doc_id=f"fk_{table_name}_{fk['from_column']}_{fk['to_table']}"
            )

    # 5. Core SQL Query Patterns for SQLite
    query_patterns = [
        ("Filter rows by category or condition: SELECT * FROM table WHERE column = 'value' AND date_col >= '2023-01-01'",
         {"type": "sql_pattern", "category": "filtering"}),
        ("Aggregate metrics: SELECT category_col, COUNT(*), SUM(amount), AVG(salary), ROUND(AVG(rating), 2) FROM table GROUP BY category_col ORDER BY SUM(amount) DESC",
         {"type": "sql_pattern", "category": "aggregation"}),
        ("Multi-table Join: SELECT t1.name, t2.department_name, t3.amount FROM t1 JOIN t2 ON t1.dept_id = t2.id JOIN t3 ON t1.id = t3.emp_id",
         {"type": "sql_pattern", "category": "join"}),
        ("Top N ranking: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5",
         {"type": "top_n"}),
        ("Date and Time operations in SQLite: strftime('%Y', date_col) for year, strftime('%Y-%m', date_col) for month, date(date_col) for date",
         {"type": "sql_pattern", "category": "datetime"}),
        ("Conditional counts and sums: SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed_count",
         {"type": "sql_pattern", "category": "conditional"}),
    ]
    for p_text, p_meta in query_patterns:
        vector_store.add_chunk(text=p_text, metadata=p_meta, doc_id=f"pattern_{p_meta.get('category', 'misc')}")

    vector_store.build_index()
