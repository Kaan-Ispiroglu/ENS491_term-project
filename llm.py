"""
Multi-provider LLM router.

  chat(prompt)       → Groq  (Llama 3.3 70B)  — sub-second latency for user-facing chat
  structured(prompt) → Mistral (mistral-small) — most reliable JSON output
  background(prompt) → Gemini Flash            — nightly/batch tasks, no latency pressure

Each function has a full fallback chain so the app keeps working if a provider
is down or its key is missing.
"""

import os
from config import gemini_client, GEMINI_MODEL

# ── Provider clients (lazy — missing keys don't crash on import) ──────────────

_groq_client    = None
_mistral_client = None


def _groq():
    global _groq_client
    if _groq_client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        from groq import Groq
        _groq_client = Groq(api_key=key)
    return _groq_client


def _mistral():
    global _mistral_client
    if _mistral_client is None:
        key = os.environ.get("MISTRAL_API_KEY", "")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY not set")
        from mistralai import Mistral
        _mistral_client = Mistral(api_key=key)
    return _mistral_client


# ── Raw completions ───────────────────────────────────────────────────────────

def _via_gemini(prompt: str) -> str:
    return gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt
    ).text.strip()


def _via_groq(prompt: str) -> str:
    resp = _groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _via_mistral(prompt: str) -> str:
    resp = _mistral().chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,      # low temp → more deterministic JSON
    )
    return resp.choices[0].message.content.strip()


# ── Public router functions ───────────────────────────────────────────────────

def chat(prompt: str) -> str:
    """
    User-facing conversational response.
    Groq → Gemini fallback.
    """
    try:
        return _via_groq(prompt)
    except Exception as e:
        print(f"[llm:chat] Groq failed ({e}), falling back to Gemini")
        return _via_gemini(prompt)


def structured(prompt: str) -> str:
    """
    Structured JSON output (gap extraction, RQ generation).
    Mistral → Groq → Gemini fallback.
    """
    try:
        return _via_mistral(prompt)
    except Exception as e:
        print(f"[llm:structured] Mistral failed ({e}), trying Groq")
    try:
        return _via_groq(prompt)
    except Exception as e:
        print(f"[llm:structured] Groq failed ({e}), falling back to Gemini")
        return _via_gemini(prompt)


def background(prompt: str) -> str:
    """
    Nightly / background tasks where latency doesn't matter.
    Groq → Mistral → Gemini fallback (Gemini free tier exhausts quickly).
    """
    try:
        return _via_groq(prompt)
    except Exception as e:
        print(f"[llm:background] Groq failed ({e}), trying Mistral")
    try:
        return _via_mistral(prompt)
    except Exception as e:
        print(f"[llm:background] Mistral failed ({e}), falling back to Gemini")
        return _via_gemini(prompt)


# ── Provider status (for diagnostics) ────────────────────────────────────────

def provider_status() -> dict:
    """Report which providers are configured. Does NOT make API calls (would burn quota)."""
    def has(key: str) -> str:
        v = os.environ.get(key, "")
        return "configured" if v and not v.startswith("your_") else "missing key"
    return {
        "groq":    has("GROQ_API_KEY"),
        "mistral": has("MISTRAL_API_KEY"),
        "gemini":  has("GEMINI_API_KEY"),
    }
