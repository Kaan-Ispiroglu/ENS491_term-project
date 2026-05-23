"""
Neo4j sidecar — optional citation/concept graph.

Soft-fails when NEO4J_* env vars are missing: every public function returns
quickly without raising, so the rest of the app keeps working on MongoDB alone.

Schema:
    (:Paper {id, title, year, field})-[:CITES]->(:Paper)
    (:Paper)-[:ABOUT]->(:Concept {name})
"""
import os

_driver = None
_checked = False


def _get_driver():
    """Lazy, cached driver. Returns None if Neo4j is not configured."""
    global _driver, _checked
    if _checked:
        return _driver
    _checked = True

    uri  = os.environ.get("NEO4J_URI", "")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    pwd  = os.environ.get("NEO4J_PASSWORD", "")

    if not uri or not pwd or pwd.startswith("your_") or "xxxxxxxx" in uri:
        return None  # not configured — silent disable

    try:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(uri, auth=(user, pwd))
        # Cheap connectivity check; raises if cluster is unreachable
        _driver.verify_connectivity()
        print("[graph] Neo4j connected.")
    except Exception as e:
        print(f"[graph] Neo4j disabled: {e}")
        _driver = None
    return _driver


def is_enabled() -> bool:
    return _get_driver() is not None


# ── Writes ────────────────────────────────────────────────────────────────────

def add_paper(paper: dict) -> None:
    """
    Best-effort: persist paper node + outgoing citation/concept edges.
    `paper` must contain at minimum {openalex_id, title, year, field}.
    Optional: referenced_works (list of openalex ids), concepts (list of names).
    """
    drv = _get_driver()
    if not drv or not paper.get("openalex_id"):
        return
    try:
        with drv.session() as s:
            s.execute_write(_write_paper_tx, paper)
    except Exception as e:
        print(f"[graph] add_paper failed: {e}")


def _write_paper_tx(tx, p: dict):
    tx.run(
        """
        MERGE (paper:Paper {id: $id})
        SET paper.title = $title, paper.year = $year, paper.field = $field
        """,
        id=p["openalex_id"], title=p.get("title", ""),
        year=p.get("year"), field=p.get("field", ""),
    )
    refs = p.get("referenced_works") or []
    if refs:
        tx.run(
            """
            UNWIND $refs AS ref
            MERGE (cited:Paper {id: ref})
            WITH cited
            MATCH (src:Paper {id: $src})
            MERGE (src)-[:CITES]->(cited)
            """,
            src=p["openalex_id"], refs=refs,
        )
    concepts = (p.get("concepts") or [])[:5]
    if concepts:
        tx.run(
            """
            UNWIND $concepts AS name
            MERGE (c:Concept {name: name})
            WITH c
            MATCH (src:Paper {id: $src})
            MERGE (src)-[:ABOUT]->(c)
            """,
            src=p["openalex_id"], concepts=concepts,
        )


# ── Reads ─────────────────────────────────────────────────────────────────────

def citation_neighbors(paper_ids: list[str], limit: int = 20) -> list[dict]:
    """
    Papers most-cited by the given set — useful for "what should we read next".
    Returns [{id, title, times_cited}, ...]. Empty list if graph disabled.
    """
    drv = _get_driver()
    if not drv or not paper_ids:
        return []
    try:
        with drv.session() as s:
            result = s.run(
                """
                MATCH (p:Paper)-[:CITES]->(cited:Paper)
                WHERE p.id IN $ids
                RETURN cited.id AS id,
                       coalesce(cited.title, '(unknown)') AS title,
                       count(p) AS times_cited
                ORDER BY times_cited DESC
                LIMIT $limit
                """,
                ids=paper_ids, limit=limit,
            )
            return [dict(r) for r in result]
    except Exception as e:
        print(f"[graph] citation_neighbors failed: {e}")
        return []


def shared_concepts(paper_ids: list[str], limit: int = 10) -> list[dict]:
    """Concepts most-shared across the given papers."""
    drv = _get_driver()
    if not drv or not paper_ids:
        return []
    try:
        with drv.session() as s:
            result = s.run(
                """
                MATCH (p:Paper)-[:ABOUT]->(c:Concept)
                WHERE p.id IN $ids
                RETURN c.name AS name, count(p) AS papers
                ORDER BY papers DESC
                LIMIT $limit
                """,
                ids=paper_ids, limit=limit,
            )
            return [dict(r) for r in result]
    except Exception as e:
        print(f"[graph] shared_concepts failed: {e}")
        return []


def stats() -> dict:
    """Counts for the health endpoint / KB panel."""
    drv = _get_driver()
    if not drv:
        return {"enabled": False, "reason": "NEO4J_* env vars not set"}
    try:
        with drv.session() as s:
            papers   = s.run("MATCH (p:Paper) RETURN count(p) AS c").single()["c"]
            cites    = s.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
            concepts = s.run("MATCH (c:Concept) RETURN count(c) AS c").single()["c"]
            return {"enabled": True, "papers": papers, "citations": cites, "concepts": concepts}
    except Exception as e:
        return {"enabled": True, "error": str(e)[:80]}
