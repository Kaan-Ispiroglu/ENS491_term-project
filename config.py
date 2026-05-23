import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-2.0-flash"

OPENALEX_EMAIL = os.environ["OPENALEX_EMAIL"]
