"""
Thin layer over `sources.py`.

Historically this module held the OpenAlex client directly. After the
data-source abstraction was added (proposal §3.2 sustainability constraint),
`fetch_papers` and `invert_abstract` live in `sources.py`. This module now
only owns the `Document` shape and the normalization to it.
"""
from dataclasses import dataclass, field

# Re-export the public source-layer API so existing imports keep working.
from sources import fetch_papers, invert_abstract  # noqa: F401


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


def papers_to_documents(papers: list[dict]) -> list[Document]:
    """Convert normalized paper dicts (from sources.fetch_papers) into Documents."""
    docs = []
    for p in papers:
        if not p.get("id"):
            continue
        docs.append(Document(
            text=f"{p.get('title','')}\n\n{p.get('abstract','')}",
            metadata={
                # `openalex_id` is kept as the key name because the rest of the
                # codebase (KB, graph) already uses it as the canonical paper id.
                "openalex_id":      p["id"],
                "title":            p.get("title", ""),
                "year":             p.get("year"),
                "doi":              p.get("doi", ""),
                "authors":          p.get("authors", []),
                "concepts":         p.get("concepts", []),
                "venue":            p.get("venue", ""),
                "citations":        p.get("citations", 0),
                "referenced_works": p.get("referenced_works", []),
                "source":           p.get("source", "openalex"),
                "reliability":      p.get("reliability", "unknown"),
            },
            doc_id=p["id"],
        ))
    return docs
