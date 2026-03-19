import requests
from llama_index.core import Document

SEED_QUERIES = [
    "persons background socioeconomic",
    "biographical background outcomes",
    "social origin life history",
    "immigration background identity",
    "cultural background educational attainment",
]

def fetch_papers(query: str, per_page: int = 50) -> list[dict]:
    """Snowball from a seed query via OpenAlex."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": per_page,
        "mailto": "your@email.com",  # polite pool
        "select": "id,title,abstract_inverted_index,authorships,"
                  "referenced_works,publication_year,doi,concepts",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json().get("results", [])


def invert_abstract(inv_index: dict | None) -> str:
    """OpenAlex stores abstracts as inverted indexes — reconstruct the text."""
    if not inv_index:
        return ""
    words = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def papers_to_documents(papers: list[dict]) -> list[Document]:
    """Convert OpenAlex results into LlamaIndex Document objects."""
    docs = []
    for p in papers:
        abstract = invert_abstract(p.get("abstract_inverted_index"))
        authors = [
            a["author"]["display_name"]
            for a in p.get("authorships", [])
            if a.get("author")
        ]
        concepts = [c["display_name"] for c in p.get("concepts", [])]

        docs.append(Document(
            text=f"{p.get('title', '')}\n\n{abstract}",
            metadata={
                "openalex_id":      p["id"],
                "title":            p.get("title", ""),
                "year":             p.get("publication_year"),
                "doi":              p.get("doi", ""),
                "authors":          authors,
                "concepts":         concepts,
                "referenced_works": p.get("referenced_works", []),
            },
            doc_id=p["id"],
        ))
    return docs