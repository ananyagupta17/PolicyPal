import os
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec  # keep PodSpec since you're already using it
from backend.services.embedding import get_embedding
from backend.services.text_chunker import chunk_text
from backend.app.document_parser import extract_text

# Load .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

INDEX_NAME = "policy-embeddings"
DIMENSION = 768  # all-mpnet-base-v2
REGION = "us-east-1"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"📦 Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=PodSpec(name="starter", environment=REGION)
    )
else:
    print(f"✅ Index '{INDEX_NAME}' already exists. Skipping creation.")

# Connect to index
index = pc.Index(INDEX_NAME)

def generate_source_id(document_url: str) -> str:
    """Stable source_id from the document URL."""
    return hashlib.md5(document_url.encode()).hexdigest()

def store_embeddings_for_text(text: str, source_id: str) -> int:
    """
    Upsert embeddings for a document's text into a namespace named after source_id.
    No cleanup; deterministic IDs so repeats overwrite.
    Returns number of vectors upserted.
    """
    chunks = chunk_text(text)
    vectors = []

    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        # Deterministic per-chunk ID so re-upserts overwrite
        vector_id = f"{i:06d}"
        embedding = get_embedding(chunk)
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i,
                "source": source_id,  # optional with namespace isolation; helpful for debugging
            }
        })

    # Namespace-per-document isolation (no deletes needed)
    index.upsert(vectors=vectors, namespace=source_id)
    print(f"✅ Stored {len(vectors)} embeddings under namespace={source_id}")
    return len(vectors)

def ingest_document(document_url: str) -> str:
    """
    Convenience: parse, embed, and upsert a document by URL.
    Returns the source_id (namespace) you should use for querying.
    """
    source_id = generate_source_id(document_url)
    text = extract_text(document_url)
    if not text or not text.strip():
        raise ValueError("No extractable text found in the document.")
    store_embeddings_for_text(text, source_id=source_id)
    return source_id
