import os
import hashlib
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/text-embedding-004"  # 768-dim, free tier
DIMENSION = 768

embedding_cache = {}

def get_embedding(text: str) -> list[float]:
    """
    Return a 768-dim embedding using Gemini text-embedding-004.
    Uses md5-based in-memory cache for repeated inputs.
    """
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]

    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    embedding = result["embedding"]
    embedding_cache[text_hash] = embedding
    return embedding