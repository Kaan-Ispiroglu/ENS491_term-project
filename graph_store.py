from llama_index.core import StorageContext, KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from config import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD


def get_graph_store() -> Neo4jGraphStore:
    return Neo4jGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL,
        database="neo4j",
    )


def build_lexical_graph(documents: list) -> KnowledgeGraphIndex:
    """
    LazyGraphRAG Step 1: build a lexical-only graph.
    max_triplets_per_chunk=0 disables LLM triplet extraction —
    we're only storing metadata relationships now.
    LLM reasoning is deferred to query time.
    """
    graph_store = get_graph_store()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=0,   # <-- LazyGraphRAG: no LLM at index time
        include_embeddings=True,    # vector search enabled for neighborhoods
        show_progress=True,
    )

    # Manually write CITED relationships (pure metadata, still no LLM)
    with graph_store._driver.session() as session:
        for doc in documents:
            openalex_id = doc.metadata["openalex_id"]
            for cited_id in doc.metadata.get("referenced_works", []):
                session.run(
                    """
                    MERGE (a:Work {id: $src})
                    MERGE (b:Work {id: $dst})
                    MERGE (a)-[:CITED]->(b)
                    """,
                    src=openalex_id, dst=cited_id,
                )
    return index