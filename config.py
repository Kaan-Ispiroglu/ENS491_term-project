import os
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

load_dotenv()

# Gemini — deferred: only called at query time (LazyGraphRAG principle)
Settings.llm = Gemini(
    model="models/gemini-1.5-flash",
    api_key=os.environ["GEMINI_API_KEY"],
)
Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001",
    api_key=os.environ["GEMINI_API_KEY"],
)

# Neo4j AuraDB
NEO4J_URL      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# OpenAlex polite pool
OPENALEX_EMAIL = os.environ["OPENALEX_EMAIL"]