from sentence_transformers import SentenceTransformer
import hashlib


model = SentenceTransformer("all-mpnet-base-v2")

embedding_cache = {}

def get_embedding(text):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]

    embedding = model.encode(text, convert_to_numpy=True).tolist()

    embedding_cache[text_hash] = embedding
    return embedding
