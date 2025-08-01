from app.services.pinecone_store import ingest_document, generate_source_id
from app.services.pipeline_qa import answer_questions
import sys

def run_pipeline_from_url(url, questions):
    try:
        print("Starting pipeline...")

        # Step 1–4: Ingest document (extract, chunk, embed, store)
        source_id = ingest_document(url)
        print(f"Document ingested. source_id / namespace: {source_id}")

        # Step 5–7: Answer questions (retrieval + LLM/fallback)
        answers = answer_questions(document_url=url, questions=questions, top_k=8)
        print({"answers": answers})

    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <document_url> [question1] [question2] ...")
        sys.exit(1)

    url = sys.argv[1]
    questions = sys.argv[2:] or [
        "Is cataract surgery covered under this policy?",
        "What is the waiting period for pre-existing diseases?",
        "What is the grace period for premium payment?"
    ]
    run_pipeline_from_url(url, questions)
