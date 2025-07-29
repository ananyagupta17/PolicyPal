from app.services.document_parser import extract_text, clean_text
from app.services.pinecone_store import store_embeddings, generate_source_id
from app.services.cleanup import delete_by_source, delete_all
import sys

def run_pipeline_from_url(url):
    try:
        print("🚀 Starting pipeline...")

        # Step 1: Extract raw text
        raw_text = extract_text(url)
        print(f"Extracted text length: {len(raw_text)}")

        # Step 2: Clean the text
        cleaned = clean_text(raw_text)
        print(f"Cleaned text length: {len(cleaned)}")

        # Step 3: Generate a unique source ID from the URL
        source_id = generate_source_id(url)
        print(f"Generated source_id: {source_id}")

        # Step 4: Store in Pinecone
        print("Storing embeddings in Pinecone...")
        store_embeddings(cleaned, source_id=source_id)
        print("Done storing in Pinecone.")

        # Step 5: Delete after use (cleanup)
        print("Deleting embeddings after processing...")
        delete_by_source(source_id)
        print(f"Deleted all embeddings with source_id: {source_id}")

        #clear storage
        # print("Deleting all embeddings")
        # delete_all()
        # print("delete all embeddings")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pipeline.py <document_url>")
    else:
        url = sys.argv[1]
        run_pipeline_from_url(url)
