from app.services.text_chunker import chunk_text
from app.services.embedding import get_embedding  
from tqdm import tqdm

def process_text_to_embeddings(text):
    chunks = chunk_text(text)
    chunk_embeddings = []

    for chunk in tqdm(chunks, desc="Embedding chunks"):
        embedding = get_embedding(chunk)
        chunk_embeddings.append({
            "text": chunk,
            "embedding": embedding
        })

    return chunk_embeddings
