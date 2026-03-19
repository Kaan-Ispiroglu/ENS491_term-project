from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import StorageContext
from graph_store import get_graph_store


def build_query_engine(index) -> KnowledgeGraphQueryEngine:
    """
    LazyGraphRAG Step 2: attach the LLM only at query time.
    retriever_mode='hybrid' uses both vector similarity and
    graph traversal to find the relevant neighborhood first —
    then Gemini reasons over only that subgraph.
    """
    return index.as_query_engine(
        include_text=True,
        retriever_mode="hybrid",       # vector + graph traversal
        response_mode="tree_summarize",
        verbose=True,
    )


# --- Example queries for persons background gap-finding ---
GAP_QUERIES = [
    "What measurement inconsistencies exist in how 'persons background' "
    "is operationalized across studies?",

    "Which under-studied populations appear in the limitations sections "
    "of persons background research?",

    "What methodological gaps are most frequently cited as future research "
    "directions in biographical background studies?",

    "Which papers act as bridges between immigration background research "
    "and psychological outcome studies?",
]