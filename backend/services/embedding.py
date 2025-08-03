import os
import json
import hashlib
import time
import logging
import urllib.request
from dotenv import load_dotenv

# optional: enable debug logging for troubleshooting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load .env if present
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError(
        "Missing COHERE_API_KEY. Ensure you have a `.env` in project root with:\n"
        "COHERE_API_KEY=your_key_here (no quotes), or export it in the shell."
    )

EMBEDDING_URL = "https://api.cohere.ai/v1/embed"
MODEL = "embed-english-v3.0"  # or "embed-multilingual-v3.0"

# Tiny in-memory cache to avoid recomputing the same chunk
embedding_cache = {}

def _call_cohere_api(text: str, input_type: str = "search_document") -> list[float]:
    """
    Call Cohere's embedding API.
    input_type must be 'search_document' or 'search_query' for v3 models.
    """
    payload = json.dumps({
        "model": MODEL,
        "texts": [text],
        "input_type": input_type
    }).encode("utf-8")

    req = urllib.request.Request(
        EMBEDDING_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                if resp.status != 200:
                    logger.warning("Cohere API returned status %s: %s", resp.status, body.decode())
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                    raise RuntimeError(f"Cohere API error {resp.status}: {body.decode()}")
                resp_json = json.loads(body)
                return resp_json["embeddings"][0]
        except urllib.error.HTTPError as e:
            body = e.read().decode() if hasattr(e, 'read') else str(e)
            if e.code in (429, 502, 503, 504) and attempt < 2:
                backoff = 2 ** attempt
                logger.warning("Transient Cohere error %s, backing off %s seconds", e.code, backoff)
                time.sleep(backoff)
                continue
            raise RuntimeError(f"Cohere API HTTP error {e.code}: {body}") from e
        except Exception as e:
            logger.warning("Network error on Cohere API call: %s", e)
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Network error calling Cohere API: {e}") from e

    raise RuntimeError("Failed to get embedding from Cohere after retries.")

def get_embedding(text: str, input_type: str = "search_document") -> list[float]:
    """
    Returns an embedding from Cohere's Embeddings API.
    input_type: 'search_document' for documents, 'search_query' for queries.
    Uses an md5-based cache for repeated inputs.
    """
    text_hash = hashlib.md5((text + input_type).encode()).hexdigest()
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]

    embedding = _call_cohere_api(text, input_type)
    embedding_cache[text_hash] = embedding
    return embedding
