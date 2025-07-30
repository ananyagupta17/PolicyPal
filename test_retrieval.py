# test_retrieval.py
"""
Smoke-test for semantic retrieval:
1) Ingest document from URL (creates/overwrites vectors for that doc).
2) Run semantic search for a few questions.
3) Print top matches (score/id/chunk/snippet).

It tries two query modes automatically:
  A) namespace="default" + filter by source_id   (single-namespace isolation)
  B) namespace=source_id (per-namespace isolation)

Use:  python test_retrieval.py --url "<doc-url>" --topk 8 --show 3
"""

import argparse
import textwrap
from typing import List, Dict

from app.services.pinecone_store import ingest_document, generate_source_id, index
from app.services.retrieval import semantic_search

# --- Default sample URL (you can override with --url) ---
DEFAULT_DOC_URL = (
    "https://hackrx.blob.core.windows.net/assets/policy.pdf"
    "?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z"
    "&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
)

DEFAULT_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
]

# --------- helpers ---------
def wrap(s: str, width: int = 120) -> str:
    return "\n       ".join(textwrap.wrap(s, width=width, replace_whitespace=False))

def show_hits(question: str, hits: List[Dict], n: int) -> None:
    print(f"\nQ: {question}")
    if not hits:
        print("  (no matches)")
        return
    for i, h in enumerate(hits[:n], 1):
        snippet = (h.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        print(f"  {i}. score={h.get('score', 0):.4f}  id={h.get('id')}  chunk={h.get('chunk_index')}")
        print(f"     {wrap(snippet)}")

def print_index_stats():
    stats = index.describe_index_stats()
    try:
        namespaces = getattr(stats, "namespaces", None) or stats.get("namespaces", {})
    except Exception:
        namespaces = {}
    total_vecs = getattr(stats, "total_vector_count", None) or stats.get("total_vector_count", None)
    print("\n--- Index Stats ---")
    print("Namespaces:", list(namespaces.keys()))
    print("Vector counts by namespace:", {k: v.get("vector_count") for k, v in namespaces.items()})
    print("Total vectors:", total_vecs)

def fetch_probe(sample_id: str, namespace: str):
    try:
        res = index.fetch(ids=[sample_id], namespace=namespace)
        print(f"Fetch probe id={sample_id!r} ns={namespace!r}: found={bool(res and res.get('vectors'))}")
    except Exception as e:
        print(f"Fetch probe failed for ns={namespace!r}: {e}")

# --------- main ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=DEFAULT_DOC_URL, help="Document URL")
    parser.add_argument("--topk", type=int, default=8, help="Top-K matches to retrieve")
    parser.add_argument("--show", type=int, default=3, help="How many matches to print per question")
    parser.add_argument("--no-ingest", action="store_true", help="Skip ingest (just query)")
    parser.add_argument("--questions", nargs="*", help="Custom questions; if omitted, uses defaults")
    args = parser.parse_args()

    doc_url = args.url
    questions = args.questions if args.questions else DEFAULT_QUESTIONS
    top_k = args.topk
    show_n = args.show

    # Compute source_id (stable per URL)
    source_id = generate_source_id(doc_url)

    # Ingest unless skipped
    if not args.no_ingest:
        print(f"Ingesting document:\n  {doc_url}")
        ns_used = ingest_document(doc_url)  # returns source_id (namespace if you're using per-namespace)
        print(f"Ingest complete. source_id={ns_used}")

    print_index_stats()
    # Quick fetch probe for two likely IDs depending on your upsert strategy
    fetch_probe(sample_id=f"{source_id}-000000", namespace="default")  # single namespace + filter strategy
    fetch_probe(sample_id="000000", namespace=source_id)               # per-namespace strategy

    # Try Strategy A: single namespace + filter (recommended MVP)
    def query_strategy_a(q: str) -> List[Dict]:
        return semantic_search(
            q,
            top_k=top_k,
            namespace="default",
            fltr={"source": {"$eq": source_id}},
            min_score=0.0,
        )

    # Fallback Strategy B: per-namespace (if you kept that)
    def query_strategy_b(q: str) -> List[Dict]:
        return semantic_search(
            q,
            top_k=top_k,
            namespace=source_id,
            fltr=None,
            min_score=0.0,
        )

    # Run questions
    for q in questions:
        hits = query_strategy_a(q)
        if not hits:
            print("  (no matches with A: default+filter) — retrying with B: per-namespace")
            hits = query_strategy_b(q)
        show_hits(q, hits, n=show_n)

    print("\nDone.\n"
          "If you consistently see '(no matches)' on both strategies:\n"
          "  • Confirm upsert path: are vectors stored under 'default' with metadata.source, or under namespace=source_id?\n"
          "  • Check index stats above (vector counts per namespace).\n"
          "  • Raise --topk to 12 while testing.\n"
          "  • Ensure the embedding dim matches index dim (768 for all-mpnet-base-v2).")

if __name__ == "__main__":
    main()
