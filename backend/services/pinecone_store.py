import os
import hashlib
import time
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec, ServerlessSpec
from pinecone.exceptions import PineconeApiException
from backend.services.embedding import get_embedding  # Gemini embedder
from backend.services.text_chunker import chunk_text
from backend.app.document_parser import extract_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

# Configurable via env: "pod" or "serverless"
DEPLOY_TYPE = os.getenv("PINECONE_DEPLOY_TYPE", "serverless").lower()
INDEX_NAME = "policy-embeddings"
REGION = "us-east-1"
BATCH_SIZE = 100

pc = Pinecone(api_key=PINECONE_API_KEY)


def _make_spec():
    if DEPLOY_TYPE == "pod":
        # Replace "starter" with your accessible pod name if different (e.g., "s1")
        return PodSpec(name="starter", environment=REGION)
    else:
        return ServerlessSpec(cloud="aws", region=REGION)


def _wait_for_index_deletion(name: str, timeout: int = 10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if name not in pc.list_indexes():
            return
        time.sleep(0.5)
    logger.warning("Timeout waiting for index %s to disappear after deletion.", name)


def _detect_embedding_dimension(sample_text: str = "dimension check", timeout_sec: float = 5.0) -> int:
    """
    Attempts to get a single embedding to determine its dimension, with a timeout.
    Falls back to a conservative default if detection fails.
    """
    default = 1536  # reasonable fallback for many Gemini embedding variants
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(get_embedding, sample_text)
        try:
            emb = future.result(timeout=timeout_sec)
            if not isinstance(emb, list):
                raise ValueError("Embedding not a list")
            dim = len(emb)
            logger.info("Detected Gemini embedding dimension: %s", dim)
            return dim
        except TimeoutError:
            logger.warning(
                "Timeout while detecting embedding dimension; falling back to default=%s", default
            )
        except Exception as e:
            logger.warning(
                "Error detecting embedding dimension (%s); falling back to default=%s", e, default
            )
    return default


# Dynamically determine dimension (non-blocking-ish with timeout)
DIMENSION = _detect_embedding_dimension()

# Ensure index exists with correct dimension
def _ensure_index():
    existing = pc.list_indexes()
    if INDEX_NAME in existing:
        existing_dim = None
        try:
            desc = pc.describe_index(INDEX_NAME)
            existing_dim = desc.get("dimension")
        except Exception:
            logger.warning("Could not get existing index description for %s", INDEX_NAME)

        if existing_dim != DIMENSION:
            logger.warning(
                "Index '%s' has dimension %s (expected %s), deleting and recreating...",
                INDEX_NAME,
                existing_dim,
                DIMENSION,
            )
            try:
                pc.delete_index(INDEX_NAME)
                _wait_for_index_deletion(INDEX_NAME)
            except Exception as e:
                logger.warning("Error deleting index %s: %s", INDEX_NAME, e)

            try:
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=DIMENSION,
                    metric="cosine",
                    spec=_make_spec(),
                )
                logger.info("Recreated index %s with correct dimension %s", INDEX_NAME, DIMENSION)
            except PineconeApiException as e:
                if getattr(e, "status", None) == 409:
                    logger.info("Index recreate race: already exists, continuing.")
                else:
                    raise
        else:
            logger.info("✅ Using existing index: %s (dimension=%s)", INDEX_NAME, existing_dim)
    else:
        logger.info("📦 Creating vector index: %s (type=%s, dimension=%s)", INDEX_NAME, DEPLOY_TYPE, DIMENSION)
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=_make_spec(),
            )
        except PineconeApiException as e:
            if getattr(e, "status", None) == 409:
                logger.info("Creation race: index already exists, continuing.")
            else:
                raise


# Bootstrap index
_ensure_index()
index = pc.Index(INDEX_NAME)


def generate_source_id(document_url: str) -> str:
    return hashlib.md5(document_url.encode("utf-8")).hexdigest()


def store_embeddings_for_text(text: str, source_id: str) -> int:
    """
    Embed and upsert a document's text into a namespace (source_id).
    Returns number of vectors upserted.
    """
    chunks = chunk_text(text)
    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        vectors = []
        for j, chunk in enumerate(batch_chunks, start=i):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            vector_id = f"{j:06d}"
            embedding = get_embedding(cleaned)
            if not isinstance(embedding, list):
                logger.warning("Embedding for chunk %s not a list; skipping", j)
                continue
            if len(embedding) != DIMENSION:
                logger.warning(
                    "Embedding dim %s mismatches expected %s for chunk %s; skipping",
                    len(embedding),
                    DIMENSION,
                    j,
                )
                continue
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": cleaned if len(cleaned) <= 2000 else cleaned[:2000],
                    "chunk_index": j,
                    "source": source_id,
                }
            })
        if not vectors:
            continue
        for attempt in range(1, 4):
            try:
                index.upsert(vectors=vectors, namespace=source_id)
                break
            except Exception as e:
                backoff = 2 ** (attempt - 1)
                logger.warning(
                    "Upsert attempt %s failed for namespace=%s: %s. Retrying in %s sec.",
                    attempt,
                    source_id,
                    e,
                    backoff,
                )
                time.sleep(backoff)
        else:
            raise RuntimeError(f"Failed to upsert into Pinecone namespace={source_id}")
        total += len(vectors)
    logger.info("✅ Stored %d embeddings under namespace=%s", total, source_id)
    return total


def ingest_document(document_url: str) -> str:
    """
    Extract text, embed, and upsert a document by URL.
    Returns the source_id / namespace.
    """
    source_id = generate_source_id(document_url)
    text = extract_text(document_url)
    if not text or not text.strip():
        raise ValueError("No extractable text found in the document.")
    store_embeddings_for_text(text, source_id=source_id)
    return source_id
