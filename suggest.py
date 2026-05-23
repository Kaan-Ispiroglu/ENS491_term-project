"""
Analyzes the current KB and uses Gemini to propose 3 expansion directions.
Called by the nightly scheduler and the manual "Refresh" button.
"""
import json
import llm
from knowledge_base import kb

SUGGEST_PROMPT = """
You manage a research knowledge base used to help academics find literature gaps and research questions.

Current knowledge base coverage:
{kb_summary}

Task: Suggest exactly 3 new research fields/topics to expand into that would:
1. Fill obvious gaps in the current coverage, OR
2. Provide useful cross-disciplinary context that enriches the existing fields

Return a JSON array only — no prose, no markdown fences.
Each item: {{"field": "<clear field name>", "topic": "<specific OpenAlex search query>", "reasoning": "<1-2 sentences on why this complements the existing KB>"}}
"""

COLD_START_PROMPT = """
You manage a research knowledge base for academic literature discovery.
The knowledge base is currently empty.

Suggest 3 foundational research fields that are broadly useful for understanding social science, education, and human development research.

Return a JSON array only — no prose, no markdown fences.
Each item: {{"field": "<clear field name>", "topic": "<specific OpenAlex search query>", "reasoning": "<1-2 sentences on why this is a good starting point>"}}
"""


def _strip_fences(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text.strip()


def generate_suggestions() -> list[dict]:
    """Ask Gemini to propose 3 KB expansion directions based on current coverage."""
    if not kb.docs:
        prompt = COLD_START_PROMPT
    else:
        fields = kb.fields

        # Sample up to 3 titles per field so Gemini understands the content depth
        samples: dict[str, list[str]] = {}
        for doc in kb.docs:
            f = doc.get("field", "unknown")
            if f not in samples:
                samples[f] = []
            if len(samples[f]) < 3:
                samples[f].append(doc.get("title", ""))

        kb_summary = "\n".join(
            f"  - {field} ({count} papers): {'; '.join(samples.get(field, [])[:2])}"
            for field, count in fields.items()
        )
        prompt = SUGGEST_PROMPT.format(kb_summary=kb_summary)

    text   = llm.background(prompt)
    parsed = json.loads(_strip_fences(text))
    return parsed if isinstance(parsed, list) else []
