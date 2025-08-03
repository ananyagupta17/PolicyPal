import os
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from backend.services.embedding import get_embedding
from backend.services.text_chunker import chunk_text
from backend.app.document_parser import extract_text

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

INDEX_NAME = "policy-embeddings"
DIMENSION = 1024  # Cohere embed-english-v3.0 output size

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists with correct settings
indexes = {idx["name"]: idx for idx in pc.list_indexes()}
if INDEX_NAME in indexes:
    desc = indexes[INDEX_NAME]
    if desc.get("dimension") != DIMENSION:
        print(f"⚠️ Index '{INDEX_NAME}' has wrong dimension ({desc.get('dimension')}), deleting...")
        pc.delete_index(INDEX_NAME)
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"✅ Using existing index: {INDEX_NAME}")
else:
    print(f"📦 Creating pure vector index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(INDEX_NAME)

def generate_source_id(document_url: str) -> str:
    """Stable source_id from the document URL."""
    return hashlib.md5(document_url.encode()).hexdigest()

def store_embeddings_for_text(text: str, source_id: str) -> int:
    """
    Upsert embeddings for a document's text into a namespace named after source_id.
    Deterministic IDs so repeats overwrite.
    """
    chunks = chunk_text(text)
    vectors = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        vector_id = f"{i:06d}"
        embedding = get_embedding(chunk, input_type="search_document")
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i,
                "source": source_id,
            }
        })

    index.upsert(vectors=vectors, namespace=source_id)
    print(f"✅ Stored {len(vectors)} embeddings under namespace={source_id}")
    return len(vectors)

def ingest_document(document_url: str) -> str:
    """
    Extract, embed, and upsert a document by URL.
    Returns the source_id (namespace) for querying.
    """
    source_id = generate_source_id(document_url)
    text = extract_text(document_url)
    if not text or not text.strip():
        raise ValueError("No extractable text found in the document.")
    store_embeddings_for_text(text, source_id=source_id)
    return source_id
