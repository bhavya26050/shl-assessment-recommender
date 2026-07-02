import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 1024
TOP_K_RETRIEVAL = 20
MAX_RECOMMENDATIONS = 10
MAX_TURNS = 8
API_TIMEOUT_SECONDS = 25
