import json
import llm

GAP_FINDER_PROMPT = """
You are a research analyst specializing in academic literature analysis.

Read the following paper abstract/text and extract:
1. Unresolved questions (things the authors say are unknown)
2. Methodological gaps (measurement issues, missing data types)
3. Under-studied populations
4. Suggested future research directions

Paper: {title}
Text: {text}
{kb_section}
Return a JSON object only — no prose, no markdown fences.
Keys: unresolved_questions, methodological_gaps, under_studied_populations, future_directions
Each key maps to a list of strings.
"""

RQ_PROMPT = """
You are a research analyst. Based on the following research gaps and papers on "{topic}", generate 5 focused, specific research questions.

Papers:
{papers}

Research gaps:
{gaps}
{kb_section}
Return a JSON array only — no prose, no markdown fences.
Each item: {{"q": "<research question>", "source": "<gap or paper that motivates it>"}}
"""


def _strip_fences(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text.strip()


def extract_gaps(title: str, conclusion_text: str, kb_context: str = "") -> dict:
    """Gap extraction — uses Mistral for reliable JSON output."""
    kb_section = f"\nRelevant background from knowledge base:\n{kb_context}\n" if kb_context else ""
    prompt = GAP_FINDER_PROMPT.format(title=title, text=conclusion_text, kb_section=kb_section)
    text = _strip_fences(llm.structured(prompt))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def generate_research_questions(topic: str, gaps: list, papers: list, kb_context: str = "") -> list:
    """RQ generation — uses Mistral for reliable JSON output."""
    paper_titles = "\n".join(f"- {p.get('title', '')}" for p in papers[:5])
    gap_titles   = "\n".join(f"- {g.get('title', '')}" for g in gaps[:6])
    kb_section   = f"\nRelevant background from knowledge base:\n{kb_context}\n" if kb_context else ""
    prompt = RQ_PROMPT.format(
        topic=topic,
        papers=paper_titles or "(none)",
        gaps=gap_titles or "(none identified)",
        kb_section=kb_section,
    )
    text = _strip_fences(llm.structured(prompt))
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        lines = [l.strip().lstrip("0123456789.-) ") for l in text.split("\n") if len(l.strip()) > 20]
        return [{"q": l, "source": "Literature analysis"} for l in lines[:5]]
