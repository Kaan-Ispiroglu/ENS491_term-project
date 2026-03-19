from llama_index.core import Settings

EDGE_TYPE_PROMPT = """
You are analyzing academic citation context for persons background research.
Given the text excerpt below, classify the citation relationship as ONE of:
- SUPPORT: the citing paper agrees with / builds on the cited work
- CONTRAST: the citing paper challenges or proposes an alternative
- QUESTION_REFUTATION: the citing paper directly refutes the cited work
- BACKGROUND: the citation is purely contextual / definitional

Text: {text}
Cited work title: {cited_title}

Respond with only the relationship type label.
"""

GAP_FINDER_PROMPT = """
You are a research analyst specializing in persons background studies
(socioeconomic origins, biographical history, cultural/immigration background).

Read the following conclusion/limitations section and extract:
1. Unresolved questions (things the authors say are unknown)
2. Methodological gaps (measurement issues, missing data types)
3. Under-studied populations
4. Suggested future research directions

Paper: {title}
Text: {text}

Return a JSON object with keys: unresolved_questions, methodological_gaps,
under_studied_populations, future_directions. Each is a list of strings.
"""


def classify_citation_edge(citing_text: str, cited_title: str) -> str:
    """Invoke Gemini to type a single CITED edge — called lazily at query time."""
    prompt = EDGE_TYPE_PROMPT.format(text=citing_text, cited_title=cited_title)
    response = Settings.llm.complete(prompt)
    label = response.text.strip().upper()
    valid = {"SUPPORT", "CONTRAST", "QUESTION_REFUTATION", "BACKGROUND"}
    return label if label in valid else "BACKGROUND"


def extract_gaps(title: str, conclusion_text: str) -> dict:
    """Run GapFinder on a single paper's conclusion section."""
    import json
    prompt = GAP_FINDER_PROMPT.format(title=title, text=conclusion_text)
    response = Settings.llm.complete(prompt)
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"raw": response.text}