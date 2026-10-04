"""
Hybrid Vector Store & Schema Indexer for SQL RAG
Supports multiple embedding backends (Ollama, SentenceTransformers, TF-IDF fallback)
with automatic failover, batch processing, and normalized cosine similarity search.
"""

import os
import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    doc_id: str = ""


class VectorStore:
    def __init__(self, backend: str = "auto", ollama_model: str = "nomic-embed-text", hf_model: str = "BAAI/bge-small-en-v1.5"):
        self.backend = backend  # 'ollama', 'sentence-transformers', 'tfidf', 'auto'
        self.ollama_model = ollama_model
        self.hf_model_name = hf_model
        self.active_backend = "tfidf"
        self._hf_model = None
        self._tfidf_vectorizer = None

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

    def add_chunk(self, text: str, metadata: Dict[str, Any], doc_id: str = ""):
        """Adds a document chunk to the vector store buffer."""
        chunk = DocumentChunk(text=text.strip(), metadata=metadata, doc_id=doc_id)
        self.chunks.append(chunk)

    def build_index(self):
        """Computes embeddings for all accumulated chunks."""
        if not self.chunks:
            self.embeddings = None
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

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> List[Tuple[DocumentChunk, float]]:
        """
        Searches the index for chunks semantically relevant to query.
        Returns list of (DocumentChunk, similarity_score).
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        if self.active_backend == "tfidf" and self._tfidf_vectorizer is not None:
            q_vec = self._tfidf_vectorizer.transform([query]).toarray()
            q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9
            q_emb = (q_vec / q_norm).astype(np.float32)
        else:
            q_emb = self._embed_texts([query])

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
         {"type": "sql_pattern", "category": "top_n"}),
        ("Date and Time operations in SQLite: strftime('%Y', date_col) for year, strftime('%Y-%m', date_col) for month, date(date_col) for date",
         {"type": "sql_pattern", "category": "datetime"}),
        ("Conditional counts and sums: SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) AS completed_count",
         {"type": "sql_pattern", "category": "conditional"}),
    ]
    for p_text, p_meta in query_patterns:
        vector_store.add_chunk(text=p_text, metadata=p_meta, doc_id=f"pattern_{p_meta['category']}")

    vector_store.build_index()
