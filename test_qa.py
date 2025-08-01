# test_qa.py

print("🐍 Script started...")

from dotenv import load_dotenv
import os

load_dotenv()
print("🔑 HF_API_KEY present:", os.getenv("HF_API_KEY") is not None)

from app.services.retrieval import semantic_search
from app.utils.prompt_builder import build_llm_prompt
from app.services.huggingface_client import call_huggingface_llm
from app.services.pinecone_store import generate_source_id

QUESTION = "What is the maternity coverage waiting period?"
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf"

if __name__ == "__main__":
    namespace = generate_source_id(DOCUMENT_URL)
    print(f"📁 Namespace: {namespace}")

    print("🔍 Retrieving context chunks...")
    chunks = semantic_search(QUESTION, top_k=6, namespace=namespace)

    if not chunks:
        print("❌ No chunks found.")
        exit()

    print(f"✅ Retrieved {len(chunks)} chunks.")
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}: {c.get('text', '')[:80]}...")

    print("🧠 Building prompt...")
    prompt = build_llm_prompt(chunks, [QUESTION])

    print("\n📜 Prompt sent to LLM:\n")
    print(prompt[:800], "...\n")  # Just to verify

    print("🚀 Calling Hugging Face LLM...")
    result = call_huggingface_llm(prompt)

    print("\n📦 LLM Response:")
    print(result)
