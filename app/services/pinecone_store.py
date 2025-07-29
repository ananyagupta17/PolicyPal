import uuid
import os
import hashlib
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, PodSpec
from app.services.embedding import get_embedding
from app.services.text_chunker import chunk_text

# Load .env 
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

INDEX_NAME = "policy-embeddings"
DIMENSION = 768  # for all-mpnet-base-v2
REGION = "us-east-1"

# ✅ Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Create index if not exists
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

# ✅ Connect to index
index = pc.Index(INDEX_NAME)

# ✅ Generate source_id from URL (same doc will produce same ID)
def generate_source_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

# ✅ Store embeddings for a given document
def store_embeddings(text: str, source_id: str = "default_doc"):
    # 1. Delete old vectors for the same source_id (avoid duplicates)
    try:
        print(f"🧹 Deleting existing vectors for source_id: {source_id}")
        index.delete(filter={"source": {"$eq": source_id}}, namespace="default")
    except Exception as e:
        print(f"⚠️ Skipping delete -possibly empty or missing namespace: {e}")

    # 2. Split text into chunks and embed
    chunks = chunk_text(text)
    vectors = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        embedding = get_embedding(chunk)
        vector_id = f"{source_id}-{i}-{uuid.uuid4().hex[:8]}"

        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": source_id,
                "chunk_index": i
            }
        })

    # 3. Upsert new vectors
    index.upsert(vectors=vectors, namespace="default")
    print(f"✅ Stored {len(vectors)} embeddings in Pinecone for source_id: {source_id}")
