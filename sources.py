"""
Data-source abstraction layer (proposal §3.2 sustainability constraint, Objective 1).

Each backend implements `fetch(query, per_page) -> list[NormalizedPaper]`.
`fetch_papers()` is the single public entry point used by the rest of the app:
it queries every enabled backend, merges and deduplicates by DOI, applies a
24-hour MongoDB cache (Risk 4 mitigation), and returns ranked, normalized
paper dictionaries.

Normalized paper shape (the contract for every downstream consumer):

    {
        "id":               str,    # source-prefixed when ambiguous
        "title":            str,
        "abstract":         str,    # already reconstructed if needed
        "year":             int|None,
        "doi":              str,
        "authors":          list[str],
        "concepts":         list[str],
        "venue":            str,
        "citations":        int,
        "type":             str,    # raw type string from the source
        "referenced_works": list[str],
        "source":           "openalex" | "semantic_scholar",
        "reliability":      "peer-reviewed" | "preprint" | "unknown",
    }
"""
import os
import time
import requests
from datetime import datetime
from typing import Iterable

from config import OPENALEX_EMAIL
from db import get_db


# ── Cache (Risk 4 mitigation) ────────────────────────────────────────────────

_CACHE_TTL_SECONDS = 24 * 3600


def _cache_col():
    col = get_db()["paper_cache"]
    # TTL index — MongoDB will auto-delete expired entries.
    try:
        col.create_index("cached_at", expireAfterSeconds=_CACHE_TTL_SECONDS)
        col.create_index([("source", 1), ("query", 1), ("per_page", 1)])
    except Exception:
        pass
    return col


def _cache_get(source: str, query: str, per_page: int):
    try:
        doc = _cache_col().find_one(
            {"source": source, "query": query, "per_page": per_page},
            {"_id": 0, "papers": 1},
        )
        return doc["papers"] if doc else None
    except Exception as e:
        print(f"[sources] cache read failed: {e}")
        return None


def _cache_put(source: str, query: str, per_page: int, papers: list[dict]):
    try:
        _cache_col().replace_one(
            {"source": source, "query": query, "per_page": per_page},
            {
                "source": source, "query": query, "per_page": per_page,
                "papers": papers, "cached_at": datetime.utcnow(),
            },
            upsert=True,
        )
    except Exception as e:
        print(f"[sources] cache write failed: {e}")


# ── Source-reliability classifier (Objective 4) ──────────────────────────────

# Hosts that publish only preprints.
_PREPRINT_HOSTS = {
    "arxiv", "biorxiv", "medrxiv", "chemrxiv", "ssrn",
    "researchsquare", "preprints.org", "osf.io", "psyarxiv",
}


def classify_reliability(*, doi: str, venue: str, paper_type: str,
                         source_url: str = "", external_ids: dict | None = None) -> str:
    """
    Heuristic preprint vs peer-reviewed classifier based on available metadata.
    Returns one of: 'peer-reviewed', 'preprint', 'unknown'.
    """
    venue_l = (venue or "").lower()
    type_l  = (paper_type or "").lower()
    url_l   = (source_url or "").lower()
    ext     = external_ids or {}

    # Explicit preprint signals
    if "preprint" in type_l:
        return "preprint"
    if any(host in venue_l for host in _PREPRINT_HOSTS):
        return "preprint"
    if any(host in url_l for host in _PREPRINT_HOSTS):
        return "preprint"
    if "ArXiv" in ext or "MAG_arxiv" in ext:
        return "preprint"

    # Peer-reviewed signals: a DOI plus a non-preprint venue is a strong indicator.
    if doi and venue and "preprint" not in venue_l:
        return "peer-reviewed"

    # OpenAlex types that correspond to peer-reviewed venues
    if type_l in {"article", "journal-article", "book-chapter", "review", "proceedings-article"}:
        if venue:
            return "peer-reviewed"

    return "unknown"


# ── OpenAlex backend ─────────────────────────────────────────────────────────

def _invert_abstract(inv_index):
    if not inv_index:
        return ""
    words = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def _openalex_fetch(query: str, per_page: int) -> list[dict]:
    url = "https://api.openalex.org/works"
    params = {
        "search":   query,
        "per-page": per_page,
        "mailto":   OPENALEX_EMAIL,
        "select":   "id,title,abstract_inverted_index,authorships,referenced_works,"
                    "publication_year,doi,concepts,cited_by_count,primary_location,type",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    out = []
    for p in r.json().get("results", []):
        if not p.get("id"):
            continue
        loc        = p.get("primary_location") or {}
        venue_obj  = loc.get("source") or {}
        venue_name = venue_obj.get("display_name") or ""
        landing    = loc.get("landing_page_url") or ""

        normalized = {
            "id":               p["id"],
            "title":            p.get("title") or "",
            "abstract":         _invert_abstract(p.get("abstract_inverted_index")),
            "year":             p.get("publication_year"),
            "doi":              p.get("doi") or "",
            "authors":          [a["author"]["display_name"]
                                 for a in p.get("authorships", [])
                                 if a.get("author")],
            "concepts":         [c["display_name"] for c in p.get("concepts", [])][:8],
            "venue":            venue_name,
            "citations":        p.get("cited_by_count", 0) or 0,
            "type":             p.get("type") or "",
            "referenced_works": p.get("referenced_works") or [],
            "source":           "openalex",
        }
        normalized["reliability"] = classify_reliability(
            doi=normalized["doi"], venue=venue_name,
            paper_type=normalized["type"], source_url=landing,
        )
        out.append(normalized)
    return out


# ── Semantic Scholar backend ─────────────────────────────────────────────────

def _s2_fetch(query: str, per_page: int) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        headers["x-api-key"] = api_key
    params = {
        "query":  query,
        "limit":  min(per_page, 100),
        "fields": "paperId,title,abstract,year,venue,authors,externalIds,"
                  "citationCount,publicationTypes,publicationVenue,references.paperId",
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        # S2 is aggressive about rate limits without an API key — one retry.
        if r.status_code == 429:
            time.sleep(1.0)
            r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[sources] semantic_scholar fetch failed: {e}")
        return []

    out = []
    for p in r.json().get("data", []) or []:
        pid = p.get("paperId")
        if not pid:
            continue
        ext       = p.get("externalIds") or {}
        pub_types = p.get("publicationTypes") or []
        venue     = p.get("venue") or ((p.get("publicationVenue") or {}).get("name") or "")

        normalized = {
            "id":               f"S2:{pid}",
            "title":            p.get("title") or "",
            "abstract":         p.get("abstract") or "",
            "year":             p.get("year"),
            "doi":              ext.get("DOI", "") or "",
            "authors":          [a.get("name", "") for a in (p.get("authors") or [])],
            "concepts":         [],
            "venue":            venue,
            "citations":        p.get("citationCount", 0) or 0,
            "type":             ",".join(pub_types),
            "referenced_works": [r.get("paperId") for r in (p.get("references") or []) if r.get("paperId")],
            "source":           "semantic_scholar",
        }
        normalized["reliability"] = classify_reliability(
            doi=normalized["doi"], venue=venue,
            paper_type=normalized["type"], external_ids=ext,
        )
        out.append(normalized)
    return out


_BACKENDS = {
    "openalex":         _openalex_fetch,
    "semantic_scholar": _s2_fetch,
}


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_papers(query: str, per_page: int = 20,
                 sources: Iterable[str] = ("openalex", "semantic_scholar"),
                 use_cache: bool = True) -> list[dict]:
    """
    Multi-source paper search.

    - Tries each backend, swallowing per-source errors so one bad provider does
      not break the request.
    - Caches each (source, query, per_page) tuple for 24 h.
    - Merges by DOI when available; prefers OpenAlex over S2 on duplicates because
      OpenAlex carries richer metadata (concepts, referenced_works, etc).
    - Final ranking: citation count desc, then year desc.
    """
    merged: dict[str, dict] = {}

    for src in sources:
        backend = _BACKENDS.get(src)
        if not backend:
            continue

        papers = _cache_get(src, query, per_page) if use_cache else None
        if papers is None:
            try:
                papers = backend(query, per_page)
            except Exception as e:
                print(f"[sources] {src} fetch failed: {e}")
                continue
            if use_cache and papers:
                _cache_put(src, query, per_page, papers)

        for p in papers:
            key = (p.get("doi") or "").lower() or p["id"]
            if key in merged:
                # Prefer OpenAlex on duplicate (more concepts + referenced_works)
                if merged[key]["source"] != "openalex" and p["source"] == "openalex":
                    merged[key] = p
            else:
                merged[key] = p

    result = sorted(
        merged.values(),
        key=lambda p: (-(p.get("citations") or 0), -(p.get("year") or 0)),
    )
    return result[:per_page]


# Backwards-compat re-export
def invert_abstract(inv_index):
    return _invert_abstract(inv_index)
