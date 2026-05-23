"""
Shared MongoDB connection.
All collections live in the "researchlens" database.
Set MONGODB_URI in .env — e.g. mongodb+srv://user:pass@cluster.mongodb.net/
"""
import os
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

_client: MongoClient | None = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        uri = os.environ.get("MONGODB_URI", "")
        if not uri:
            raise RuntimeError("MONGODB_URI not set in .env")
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    return _get_client()["researchlens"]


def kb_col() -> Collection:
    col = get_db()["knowledge_base"]
    # Unique index so re-ingesting the same paper is a no-op
    col.create_index([("openalex_id", ASCENDING)], unique=True, sparse=True)
    return col


def suggestions_col() -> Collection:
    col = get_db()["suggestions"]
    col.create_index([("id", ASCENDING)], unique=True)
    return col


def usage_col() -> Collection:
    """One document per calendar date: {date: "YYYY-MM-DD", papers_ingested: N}"""
    col = get_db()["daily_usage"]
    col.create_index([("date", ASCENDING)], unique=True)
    return col
