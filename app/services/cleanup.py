# cleanup.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "policy-embeddings"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def delete_by_source(source_id: str):
    print(f"Deleting vectors for source_id: {source_id}")
    index.delete(filter={"source": {"$eq": source_id}}, namespace="default")
    print(f"Deleted all vectors with source_id: {source_id}")


def delete_all():
    print("⚠️ Deleting ALL vectors in the index!")
    index.delete(delete_all=True)
    print("Index is now empty.")

# Example usage:
if __name__ == "__main__":
    # Change this to test
    delete_by_source("test_doc")
    # Or uncomment this to wipe everything
    # delete_all()
