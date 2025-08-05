import os
import hashlib
import time
import logging
from typing import List
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_EMBD_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_EMBD_KEY in .env")

MODEL = "models/embedding-001"
BATCH_EMBED_URL = f"https://generativelanguage.googleapis.com/v1/{MODEL}:batchEmbedContents"

# Simple in-memory cache
_embedding_cache: dict[str, List[float]] = {}

MAX_BATCH_SIZE = 250  # Gemini API limit


def _call_gemini_batch_api(
    texts: List[str], max_retries: int = 3, delay_base: float = 1.0
) -> List[List[float]]:
    """
    Call Gemini batch embedding API for a list of texts.
    Handles both wrapped and raw embedding formats.
    """
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    payload = {
        "requests": [
            {"model": MODEL, "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                BATCH_EMBED_URL, headers=headers, params=params, json=payload, timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings_raw = data.get("embeddings", [])

                embeddings_list: List[List[float]] = []
                for idx, item in enumerate(embeddings_raw):
                    if isinstance(item, dict):
                        if "embedding" in item and "values" in item["embedding"]:
                            embeddings_list.append(item["embedding"]["values"])
                        elif "values" in item:
                            embeddings_list.append(item["values"])
                        else:
                            logger.error("Unexpected embedding dict format: %s", item)
                            raise RuntimeError("Invalid embedding response format.")
                    elif isinstance(item, list):
                        embeddings_list.append(item)
                    else:
                        logger.error("Unexpected embedding type: %s", type(item))
                        raise RuntimeError("Invalid embedding response format.")

                return embeddings_list

            elif resp.status_code in (429, 502, 503, 504) and attempt < max_retries:
                backoff = delay_base * (2 ** (attempt - 1))
                logger.warning("Transient Gemini error %s. Retrying in %.1f sec",
                               resp.status_code, backoff)
                time.sleep(backoff)
                continue
            else:
                logger.error("Gemini API error %s: %s", resp.status_code, resp.text)
                resp.raise_for_status()
        except requests.RequestException as e:
            if attempt < max_retries:
                backoff = delay_base * (2 ** (attempt - 1))
                logger.warning("Network error on Gemini API call: %s; retrying in %.1f sec",
                               e, backoff)
                time.sleep(backoff)
                continue
            raise RuntimeError(f"Failed to call Gemini embedding API: {e}") from e

    raise RuntimeError("Exceeded retries calling Gemini embedding API.")


def get_embedding(text: str) -> List[float]:
    """
    Get a single embedding (with caching) by calling the batch API for one text.
    """
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if key in _embedding_cache:
        return _embedding_cache[key]

    embedding = get_embeddings([text])[0]
    _embedding_cache[key] = embedding
    return embedding


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using batching and caching.
    """
    results: List[List[float]] = []
    to_fetch: List[str] = []
    fetch_indices: List[int] = []

    # Pull from cache
    for idx, text in enumerate(texts):
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in _embedding_cache:
            results.append(_embedding_cache[key])
        else:
            results.append(None)  # placeholder
            to_fetch.append(text)
            fetch_indices.append(idx)

    # Call API in batches for missing items
    for i in range(0, len(to_fetch), MAX_BATCH_SIZE):
        batch = to_fetch[i: i + MAX_BATCH_SIZE]
        batch_embeddings = _call_gemini_batch_api(batch)
        for j, emb in enumerate(batch_embeddings):
            orig_idx = fetch_indices[i + j]
            key = hashlib.md5(texts[orig_idx].encode("utf-8")).hexdigest()
            _embedding_cache[key] = emb
            results[orig_idx] = emb

    return results
