from ingest import fetch_papers, papers_to_documents, SEED_QUERIES
from graph_store import build_lexical_graph
from query import build_query_engine, GAP_QUERIES
from analyze import extract_gaps
import json

def main():
    # 1. Ingest — no LLM calls
    print("Fetching papers from OpenAlex...")
    all_papers = []
    for q in SEED_QUERIES:
        all_papers.extend(fetch_papers(q, per_page=50))

    # Deduplicate by OpenAlex ID
    seen = set()
    unique_papers = [p for p in all_papers
                     if p["id"] not in seen and not seen.add(p["id"])]
    print(f"Fetched {len(unique_papers)} unique papers")

    documents = papers_to_documents(unique_papers)

    # 2. Build lexical graph — still no LLM calls
    print("Building lexical graph in Neo4j (LazyGraphRAG stage 1)...")
    index = build_lexical_graph(documents)

    # 3. Extract gaps from top papers — LLM invoked here, selectively
    print("Running GapFinder on top 20 papers...")
    gap_results = []
    for doc in documents[:20]:
        if len(doc.text) > 200:  # skip near-empty abstracts
            gaps = extract_gaps(doc.metadata["title"], doc.text)
            gap_results.append({"paper": doc.metadata["title"], "gaps": gaps})

    with open("gap_report.json", "w") as f:
        json.dump(gap_results, f, indent=2)
    print("Gap report saved to gap_report.json")

    # 4. Query engine — LLM reasons over subgraphs on demand
    print("Initializing LazyGraphRAG query engine...")
    engine = build_query_engine(index)

    for query in GAP_QUERIES:
        print(f"\nQ: {query}")
        response = engine.query(query)
        print(f"A: {response}\n{'—'*60}")


if __name__ == "__main__":
    main()

#**Key things to keep in mind for your `.env` file:**

###GEMINI_API_KEY=your_key_here
###NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
###NEO4J_USERNAME=neo4j
###NEO4J_PASSWORD=your_aura_password
###OPENALEX_EMAIL=your@email.com