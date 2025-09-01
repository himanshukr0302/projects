import os
from dotenv import load_dotenv

load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Models / embeddings
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
GEMINI_CHAT_MODEL = "gemini-2.5-pro"

# Paths
INDEX_PATH = "faiss_index"
DB_PATH = "studybuddy.db"

# Quiz JSON instructions (kept strict)
QUIZ_JSON_INSTRUCTIONS = """
Return a strict JSON object with this schema: {
  "questions": [
    {
      "id": string,
      "question": string,
      "options": [string, string, string, string],
      "answer_index": integer,
      "explanation": string
    }
  ]
}
""".strip()
