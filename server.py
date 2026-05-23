import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

import config  # noqa: F401 — loads env and configures Gemini
from config import gemini_client, GEMINI_MODEL
import llm
import graph

from ingest import fetch_papers, papers_to_documents, invert_abstract
from analyze import extract_gaps, generate_research_questions
from knowledge_base import kb
from scheduler import (
    create_scheduler, nightly_suggest,
    load_suggestions, update_suggestion_status, get_suggestion,
    add_daily_usage, daily_remaining, MAX_DAILY_PAPERS,
)

_HERE = Path(__file__).parent


# ── App lifespan (scheduler start/stop) ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = create_scheduler()
    scheduler.start()
    print(f"[server] Scheduler started. Daily cap: {MAX_DAILY_PAPERS} papers. Next run: 02:00 UTC.")
    yield
    scheduler.shutdown(wait=False)
    print("[server] Scheduler stopped.")


app = FastAPI(lifespan=lifespan)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/")
async def serve_frontend():
    return FileResponse(_HERE / "index.html")


# ── Request models ────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    topic: str

class QueryRequest(BaseModel):
    message: str
    papers: list[dict] = []
    gaps: list[dict] = []
    history: list[dict] = []   # recent prior turns: [{"role":"user"|"bot","text":"..."}, ...]

class KBIngestRequest(BaseModel):
    field: str
    topic: str
    per_page: int = 15


# ── /api/analyze ──────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    async def stream():
        loop = asyncio.get_event_loop()

        yield sse({"type": "status", "message": f'Searching papers on "{req.topic}"…'})

        try:
            papers_raw = await loop.run_in_executor(None, fetch_papers, req.topic, 20)
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
            return

        if not papers_raw:
            yield sse({"type": "error", "message": "No papers found. Try a different topic."})
            return

        unique    = list({p["id"]: p for p in papers_raw if p.get("id")}.values())
        documents = papers_to_documents(unique)

        papers_out = []
        for doc in documents[:10]:
            m          = doc.metadata
            abstract_t = doc.text.split("\n\n", 1)[1][:300] if "\n\n" in doc.text else ""
            authors    = m["authors"]
            author_str = (", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")) if authors else "Unknown"
            reliability = m.get("reliability", "unknown")
            # Frontend tag check is `type === 'peer'`, so map the new
            # three-valued classification to a two-valued legacy field.
            legacy_type = "peer" if reliability == "peer-reviewed" else "preprint"
            papers_out.append({
                "id":          m["openalex_id"],
                "title":       m["title"] or "Untitled",
                "authors":     author_str,
                "year":        m["year"] or "",
                "venue":       m.get("venue") or m.get("source", "Unknown"),
                "type":        legacy_type,
                "reliability": reliability,
                "source":      m.get("source", "openalex"),
                "citations":   m.get("citations", 0),
                "abstract":    abstract_t,
            })

        yield sse({"type": "papers", "papers": papers_out})

        kb_docs    = kb.search(req.topic, k=5)
        kb_context = "\n".join(
            f"- {d['title']} ({d.get('year', '')}): {d['abstract'][:200]}" for d in kb_docs
        )
        if kb_docs:
            yield sse({"type": "kb_used", "count": len(kb_docs),
                       "titles": [d["title"] for d in kb_docs]})

        yield sse({"type": "status", "message": "Extracting research gaps in parallel…"})

        # Run all gap-extraction LLM calls concurrently — much faster than sequential.
        targets = [d for d in documents[:5] if len(d.text) > 200]
        async def extract_one(doc):
            try:
                return doc, await loop.run_in_executor(
                    None, extract_gaps, doc.metadata["title"], doc.text, kb_context
                )
            except Exception:
                return doc, None

        results = await asyncio.gather(*[extract_one(d) for d in targets])

        all_gaps: list = []
        for doc, result in results:
            if not result or "raw" in result:
                continue
            for category, items in result.items():
                if not isinstance(items, list):
                    continue
                for item in items[:2]:
                    if not item:
                        continue
                    gap = {
                        "label":   f"Gap: {category.replace('_', ' ').title()}",
                        "title":   item[:80],
                        "desc":    item,
                        "sources": f"From: {doc.metadata['title'][:60]}",
                    }
                    all_gaps.append(gap)
                    yield sse({"type": "gap", "gap": gap})

        yield sse({"type": "status", "message": "Generating research questions…"})

        try:
            rqs = await loop.run_in_executor(
                None, generate_research_questions, req.topic, all_gaps, papers_out, kb_context
            )
            for i, rq in enumerate(rqs):
                yield sse({"type": "rq", "rq": {**rq, "n": i + 1}})
        except Exception:
            pass

        yield sse({"type": "done"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /api/query ────────────────────────────────────────────────────────────────

@app.post("/api/query")
async def query(req: QueryRequest):
    async def stream():
        loop      = asyncio.get_event_loop()
        kb_docs   = kb.search(req.message, k=3)
        kb_section = (
            "\n\nRelevant background from knowledge base:\n"
            + "\n".join(f"- {d['title']} ({d.get('year','')}): {d['abstract'][:200]}" for d in kb_docs)
        ) if kb_docs else ""

        # Recent conversation turns so follow-ups make sense.
        history_lines = []
        for turn in req.history[-6:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            txt = (turn.get("text") or "").strip().replace("\n", " ")
            if txt:
                history_lines.append(f"{role}: {txt[:400]}")
        history_block = ("\n\nRecent conversation:\n" + "\n".join(history_lines)) if history_lines else ""

        prompt = f"""You are a research assistant helping a user explore the academic literature.

Papers in this session:
{chr(10).join(f"- {p['title']} ({p.get('year','')})" for p in req.papers[:5]) or '(none)'}

Known research gaps:
{chr(10).join(f"- {g['title']}" for g in req.gaps[:5]) or '(none)'}
{kb_section}{history_block}

The user's new message may be a follow-up that refers to things discussed above (e.g. "tell me more", "the second one", "why?"). Use the conversation history to resolve such references.

Answer the user's question concisely. Use **bold** for key terms. Keep under 150 words.

User: {req.message}"""

        try:
            response = await loop.run_in_executor(None, llm.chat, prompt)
            yield sse({"type": "text", "text": response})
        except Exception as e:
            yield sse({"type": "text", "text": f"Sorry, I ran into an error: {e}"})

        yield sse({"type": "done"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /api/kb — knowledge base management ──────────────────────────────────────

@app.get("/api/kb/fields")
async def kb_fields():
    return {
        "fields":          kb.fields,
        "total":           len(kb.docs),
        "daily_used":      MAX_DAILY_PAPERS - daily_remaining(),
        "daily_remaining": daily_remaining(),
        "daily_cap":       MAX_DAILY_PAPERS,
    }


@app.post("/api/kb/ingest")
async def kb_ingest(req: KBIngestRequest):
    async def stream():
        loop = asyncio.get_event_loop()

        yield sse({"type": "status", "message": f'Fetching papers on "{req.topic}"…'})

        try:
            papers_raw = await loop.run_in_executor(None, fetch_papers, req.topic, req.per_page)
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
            return

        existing   = kb.known_ids
        new_papers = [p for p in papers_raw if p["id"] not in existing]

        if not new_papers:
            yield sse({"type": "done", "ingested": 0, "message": "All papers already in KB."})
            return

        yield sse({"type": "status",
                   "message": f"Indexing {len(new_papers)} new papers…"})

        ingested = 0
        for p in new_papers:
            title    = p.get("title") or ""
            abstract = p.get("abstract", "")  # already inverted by sources layer
            if not title:
                continue
            try:
                await loop.run_in_executor(None, kb.add, req.field, title, abstract, {
                    "openalex_id":      p["id"],
                    "year":             p.get("year"),
                    "doi":              p.get("doi", ""),
                    "referenced_works": p.get("referenced_works", []),
                    "concepts":         p.get("concepts", [])[:5],
                    "source":           p.get("source", "openalex"),
                    "reliability":      p.get("reliability", "unknown"),
                })
                ingested += 1
                yield sse({"type": "progress", "ingested": ingested,
                           "total": len(new_papers), "title": title[:70]})
            except Exception as e:
                yield sse({"type": "warn", "message": f"Skipped: {str(e)[:60]}"})

        yield sse({"type": "done", "ingested": ingested, "field": req.field})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/kb/fields/{field}")
async def kb_delete_field(field: str):
    removed = kb.delete_field(field)
    return {"removed": removed, "field": field}


# ── /api/kb/suggestions ───────────────────────────────────────────────────────

@app.get("/api/kb/suggestions")
async def get_suggestions():
    pending = [s for s in load_suggestions() if s.get("status") == "pending"]
    return {"suggestions": pending, "daily_remaining": daily_remaining()}


@app.post("/api/kb/suggestions/refresh")
async def refresh_suggestions():
    """Manually trigger suggestion generation (same logic as nightly job)."""
    async def stream():
        yield sse({"type": "status", "message": "Analyzing KB coverage with Gemini…"})
        try:
            await nightly_suggest()
            pending = [s for s in load_suggestions() if s.get("status") == "pending"]
            yield sse({"type": "done", "suggestions": pending})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/kb/suggestions/{suggestion_id}/approve")
async def approve_suggestion(suggestion_id: str):
    """Approve a suggestion and ingest its papers (respects daily cap)."""
    suggestion = get_suggestion(suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    async def stream():
        loop      = asyncio.get_event_loop()
        remaining = daily_remaining()

        if remaining <= 0:
            yield sse({"type": "error",
                       "message": f"Daily cap of {MAX_DAILY_PAPERS} papers reached. Try again tomorrow."})
            return

        per_page = min(15, remaining)
        yield sse({"type": "status",
                   "message": f'Fetching up to {per_page} papers on "{suggestion["topic"]}"…'})

        try:
            papers_raw = await loop.run_in_executor(
                None, fetch_papers, suggestion["topic"], per_page
            )
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
            return

        existing   = kb.known_ids
        new_papers = [p for p in papers_raw if p["id"] not in existing]

        if not new_papers:
            update_suggestion_status(suggestion_id, "approved")
            yield sse({"type": "done", "ingested": 0,
                       "message": "All papers already in KB.", "field": suggestion["field"]})
            return

        yield sse({"type": "status", "message": f"Embedding {len(new_papers)} papers…"})

        ingested = 0
        for p in new_papers:
            title    = p.get("title") or ""
            abstract = p.get("abstract", "")  # already inverted by sources layer
            if not title:
                continue
            try:
                await loop.run_in_executor(None, kb.add, suggestion["field"], title, abstract, {
                    "openalex_id":      p["id"],
                    "year":             p.get("year"),
                    "doi":              p.get("doi", ""),
                    "referenced_works": p.get("referenced_works", []),
                    "concepts":         p.get("concepts", [])[:5],
                    "source":           p.get("source", "openalex"),
                    "reliability":      p.get("reliability", "unknown"),
                })
                ingested += 1
                add_daily_usage(1)
                yield sse({"type": "progress", "ingested": ingested,
                           "total": len(new_papers), "title": title[:70]})
            except Exception as e:
                yield sse({"type": "warn", "message": f"Skipped: {str(e)[:60]}"})

        update_suggestion_status(suggestion_id, "approved")
        yield sse({"type": "done", "ingested": ingested, "field": suggestion["field"],
                   "daily_remaining": daily_remaining()})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/kb/suggestions/{suggestion_id}")
async def dismiss_suggestion(suggestion_id: str):
    found = update_suggestion_status(suggestion_id, "dismissed")
    if not found:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    return {"dismissed": suggestion_id}


@app.get("/api/llm/status")
async def llm_status():
    """Which providers are configured and reachable."""
    loop = asyncio.get_event_loop()
    status = await loop.run_in_executor(None, llm.provider_status)
    return status


@app.get("/api/health")
async def health():
    """Liveness probe — KB count, Mongo reachability, provider config, graph stats."""
    mongo = "ok"
    kb_count = 0
    try:
        kb_count = len(kb.docs)
    except Exception as e:
        mongo = f"down: {str(e)[:80]}"
    return {
        "status":    "ok" if mongo == "ok" else "degraded",
        "mongo":     mongo,
        "kb_docs":   kb_count,
        "providers": llm.provider_status(),
        "graph":     graph.stats(),
    }


@app.get("/api/graph/neighbors")
async def graph_neighbors(ids: str, limit: int = 20):
    """
    Citation neighbours of the given papers (comma-separated OpenAlex ids).
    Returns [] if Neo4j is disabled.
    """
    id_list = [s.strip() for s in ids.split(",") if s.strip()]
    loop = asyncio.get_event_loop()
    neighbours = await loop.run_in_executor(None, graph.citation_neighbors, id_list, limit)
    concepts   = await loop.run_in_executor(None, graph.shared_concepts,   id_list, 10)
    return {"enabled": graph.is_enabled(), "neighbors": neighbours, "shared_concepts": concepts}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
