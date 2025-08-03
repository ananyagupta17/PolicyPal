from typing import List, Dict, Optional
from backend.services.embedding import get_embedding
from backend.services.pinecone_store import index

def semantic_search(
    question: str,
    top_k: int = 5,
    namespace: str = "default",
    fltr: Optional[Dict] = None,
    min_score: float = 0.0,
) -> List[Dict]:
    """
    Semantic search over Pinecone for a natural-language question.
    IMPORTANT: pass the correct 'namespace' = source_id to isolate per document.
    """
    # 1) Embed the query
    q_vec = get_embedding(question)

    # 2) Query Pinecone
    res = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        namespace=namespace,
        filter=fltr,
    )

    # 3) Normalize response across client versions
    matches = getattr(res, "matches", None) or res.get("matches", [])

    # 4) Shape results for downstream LLM
    results: List[Dict] = []
    for m in matches:
        m_id  = getattr(m, "id", None)     if not isinstance(m, dict) else m.get("id")
        score = getattr(m, "score", 0.0)    if not isinstance(m, dict) else float(m.get("score", 0.0))
        meta  = getattr(m, "metadata", {})  if not isinstance(m, dict) else m.get("metadata", {})

        if score is None or score < min_score:
            continue

        results.append({
            "id": m_id,
            "score": float(score),
            "text": meta.get("text", ""),
            "chunk_index": meta.get("chunk_index"),
            "source": meta.get("source"),
            # "page": meta.get("page"),  # add later if you track pages
        })

    return results
