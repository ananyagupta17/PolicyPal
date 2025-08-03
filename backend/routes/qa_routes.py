from fastapi import APIRouter
from typing import List, Dict, Optional
from pydantic import BaseModel

from backend.services.retrieval import semantic_search
from backend.services.qa import answer_one_question

router = APIRouter()

# --- Request/response schemas ---

class SearchRequest(BaseModel):
    question: str
    top_k: int = 5
    namespace: str = "default"
    fltr: Optional[Dict] = None

class SearchResponse(BaseModel):
    chunks: List[Dict]

class QARequest(BaseModel):
    question: str
    chunks: List[Dict]

class QAResponse(BaseModel):
    answer: str


# --- Routes ---

@router.post("/semantic-search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):
    results = semantic_search(
        question=request.question,
        top_k=request.top_k,
        namespace=request.namespace,
        fltr=request.fltr,
    )
    return {"chunks": results}


@router.post("/heuristic-answer", response_model=QAResponse)
def answer_endpoint(request: QARequest):
    ans = answer_one_question(request.question, request.chunks)
    return {"answer": ans}
