"""
MongoDB-backed knowledge base with TF-IDF search (scikit-learn).

Each document stored in MongoDB has:
  field, title, abstract, _text, openalex_id (optional), year, doi
TF-IDF index is rebuilt in-memory on each search call — fast for thousands of docs
and avoids any serialisation of scipy sparse matrices.
"""
import numpy as np
from pymongo.errors import DuplicateKeyError

import graph
from db import kb_col


class KnowledgeBase:

    def __init__(self):
        # TF-IDF cache: (vectorizer, doc_matrix, pool_docs, cache_key)
        # cache_key = (field, count) — invalidated when count changes or field filter differs.
        self._cache: dict = {}

    def _invalidate_cache(self) -> None:
        self._cache.clear()

    # ── Read helpers ──────────────────────────────────────────────────────────

    @property
    def docs(self) -> list[dict]:
        """Return all documents as plain dicts (no Mongo _id)."""
        return list(kb_col().find({}, {"_id": 0}))

    @property
    def fields(self) -> dict[str, int]:
        """Field name → paper count."""
        pipeline = [{"$group": {"_id": "$field", "count": {"$sum": 1}}}]
        return {r["_id"]: r["count"] for r in kb_col().aggregate(pipeline)}

    @property
    def known_ids(self) -> set[str]:
        return {
            d["openalex_id"]
            for d in kb_col().find(
                {"openalex_id": {"$exists": True}}, {"_id": 0, "openalex_id": 1}
            )
        }

    # ── Write helpers ─────────────────────────────────────────────────────────

    def add(self, field: str, title: str, abstract: str, meta: dict | None = None) -> None:
        """
        Insert a paper.  Silently ignores duplicates (same openalex_id).
        """
        doc = {
            "field":    field,
            "title":    title,
            "abstract": abstract[:600],
            "_text":    f"{title} {abstract}"[:1200],
            **(meta or {}),
        }
        try:
            kb_col().insert_one(doc)
            self._invalidate_cache()
        except DuplicateKeyError:
            pass  # already in KB — skip

        # Mirror to Neo4j sidecar (best-effort, silent if disabled).
        graph.add_paper(doc)

    def delete_field(self, field: str) -> int:
        result = kb_col().delete_many({"field": field})
        if result.deleted_count:
            self._invalidate_cache()
        return result.deleted_count

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5, field: str | None = None) -> list[dict]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Cache the fitted vectorizer + doc matrix per (field, count). Only the
        # query vector is recomputed on each call — the expensive corpus fit is
        # done once per KB update.
        col = kb_col()
        count = col.count_documents({"field": field} if field else {})
        if count == 0:
            return []
        cache_key = (field, count)

        cached = self._cache.get(cache_key)
        if cached is None:
            pool = list(col.find({"field": field} if field else {}, {"_id": 0}))
            corpus = [d.get("_text", d.get("title", "")) for d in pool]
            try:
                vec = TfidfVectorizer(max_features=8000, stop_words="english")
                doc_mat = vec.fit_transform(corpus)
            except ValueError:
                return []
            cached = (vec, doc_mat, pool)
            # Only keep the latest cache_key per field to avoid unbounded growth.
            self._cache = {cache_key: cached}

        vec, doc_mat, pool = cached
        try:
            q_vec = vec.transform([query])
        except ValueError:
            return []
        sims = (doc_mat @ q_vec.T).toarray().flatten()
        order = np.argsort(sims)[::-1]
        return [{**pool[i], "score": float(sims[i])} for i in order[:k]]


kb = KnowledgeBase()
