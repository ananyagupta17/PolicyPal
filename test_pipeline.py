import sys
import os

# Ensure the app/ directory is in Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.services.document_parser import extract_text, clean_text
from app.services.text_chunker import chunk_text
from app.services.embedding import get_embedding
from app.services.embedder import process_text_to_embeddings


def main(doc_url):
    print("📥 Downloading and extracting text...")
    raw_text = extract_text(doc_url)

    print("\n🧹 Cleaning text...")
    cleaned = clean_text(raw_text)
    print(f"✅ Cleaned text length: {len(cleaned)} characters\n")

    print("🔪 Chunking text...")
    chunks = chunk_text(cleaned)
    avg_len = sum(len(c) for c in chunks) // len(chunks)
    print(f"✅ Chunked into {len(chunks)} chunks | Avg chunk length: {avg_len} characters\n")

    print("🔎 Generating embeddings...")
    embeddings = process_text_to_embeddings(cleaned)
    print(f"\n✅ Generated embeddings for {len(embeddings)} chunks.")

    print("\n📌 Sample output:")
    for i, chunk_data in enumerate(embeddings[:2]):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk_data['text'][:150]}...")
        print(f"Embedding (first 5 dims): {chunk_data['embedding'][:5]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pipeline.py <document_url>")
        sys.exit(1)

    document_url = sys.argv[1]
    main(document_url)
