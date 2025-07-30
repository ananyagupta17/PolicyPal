from sentence_transformers import SentenceTransformer
import hashlib

# 768-dim embeddings for all-mpnet-base-v2
model = SentenceTransformer("all-mpnet-base-v2")

# Tiny in-memory cache to avoid recomputing the same chunk
embedding_cache = {}

def get_embedding(text: str) -> list[float]:
    """
    Return a 768-dim embedding for the given text (as Python list[float]).
    Uses an md5-based cache for repeated inputs.
    """
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]
    embedding = model.encode(text, convert_to_numpy=True).tolist()
    embedding_cache[text_hash] = embedding
    return embedding
