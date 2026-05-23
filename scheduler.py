"""
Nightly KB expansion scheduler — MongoDB-backed.

- Runs at 02:00 UTC daily
- Generates suggestions via LLM (Groq → Mistral → Gemini)
- Enforces a MAX_DAILY_PAPERS cap to stay within free-tier API limits
"""
import uuid
from datetime import date

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pymongo import ReturnDocument

from db import suggestions_col, usage_col

MAX_DAILY_PAPERS = 30


# ── Daily usage tracking ──────────────────────────────────────────────────────

def get_daily_usage() -> int:
    today = str(date.today())
    doc = usage_col().find_one({"date": today}, {"_id": 0})
    return doc["papers_ingested"] if doc else 0


def add_daily_usage(n: int) -> int:
    """Increment today's counter; returns new total."""
    today = str(date.today())
    result = usage_col().find_one_and_update(
        {"date": today},
        {"$inc": {"papers_ingested": n}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return result["papers_ingested"]


def daily_remaining() -> int:
    return max(0, MAX_DAILY_PAPERS - get_daily_usage())


# ── Suggestion persistence ────────────────────────────────────────────────────

def load_suggestions() -> list[dict]:
    return list(suggestions_col().find({}, {"_id": 0}))


def add_suggestions(new: list[dict]) -> int:
    """Merge new suggestions, skipping fields already pending. Returns count added."""
    pending_fields = {
        s["field"]
        for s in suggestions_col().find({"status": "pending"}, {"_id": 0, "field": 1})
    }
    added = 0
    for s in new:
        if s.get("field") in pending_fields:
            continue
        suggestions_col().insert_one({
            "id":          uuid.uuid4().hex[:8],
            "field":       s["field"],
            "topic":       s["topic"],
            "reasoning":   s.get("reasoning", ""),
            "proposed_at": str(date.today()),
            "status":      "pending",
        })
        added += 1
    return added


def update_suggestion_status(suggestion_id: str, status: str) -> bool:
    result = suggestions_col().update_one(
        {"id": suggestion_id}, {"$set": {"status": status}}
    )
    return result.matched_count > 0


def get_suggestion(suggestion_id: str) -> dict | None:
    return suggestions_col().find_one({"id": suggestion_id}, {"_id": 0})


# ── Nightly job ───────────────────────────────────────────────────────────────

async def nightly_suggest():
    """Generate and persist new KB expansion suggestions."""
    print("[scheduler] Running nightly suggestion job…")
    try:
        from suggest import generate_suggestions
        proposals = generate_suggestions()
        n = add_suggestions(proposals)
        print(f"[scheduler] Added {n} new suggestion(s).")
    except Exception as e:
        print(f"[scheduler] Suggestion job failed: {e}")


# ── Scheduler factory ─────────────────────────────────────────────────────────

def create_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(nightly_suggest, CronTrigger(hour=2, minute=0))
    return scheduler
