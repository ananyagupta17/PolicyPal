import re
from typing import List, Dict

# Simple keyword-ish scoring to pick relevant lines from chunks
def _score_line(q_tokens: set, line: str) -> float:
    text = line.lower()
    score = 0.0
    # overlap with query tokens
    for t in q_tokens:
        if len(t) > 2 and t in text:
            score += 1.0
    # presence of numbers often matters in policies
    if re.search(r"\b\d+\b", text):
        score += 0.3
    return score

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def answer_one_question(question: str, retrieved: List[Dict], max_lines: int = 2) -> str:
    if not retrieved:
        return "Could not find relevant information in the document."

    # Combine texts of retrieved chunks
    corpus = "\n".join([r.get("text", "") for r in retrieved if r.get("text")])
    if not corpus:
        return "Could not find relevant information in the document."

    q_tokens = set(_tokenize(question))
    # Split into candidate sentences/clauses
    lines = [ln.strip() for ln in re.split(r"[.\n]", corpus) if ln.strip()]
    if not lines:
        return "Could not find relevant information in the document."

    scored = [(ln, _score_line(q_tokens, ln)) for ln in lines]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [ln for ln, sc in scored[:max_lines] if sc > 0]

    if not top:
        # fallback to first chunk
        return retrieved[0].get("text", "")[:300] + ("..." if len(retrieved[0].get("text", "")) > 300 else "")

    answer = "; ".join(top)
    # tidy whitespace
    answer = re.sub(r"\s{2,}", " ", answer).strip()
    return answer
