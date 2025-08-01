from fastapi import APIRouter, HTTPException, Header
from typing import List
from pydantic import BaseModel
import os

from app.services.pipeline_qa import answer_questions
from app.services.pinecone_store import ingest_document

router = APIRouter()
BEARER_TOKEN = os.getenv("BEARER_TOKEN") 

class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

class DocumentResponse(BaseModel):
    answers: List[str]

@router.post("/process-document", response_model=DocumentResponse)
async def process_document(request: DocumentRequest):
    try:
        ingest_document(request.documents)
        answers = answer_questions(
            document_url=request.documents,
            questions=request.questions,
            top_k=8
        )
        return DocumentResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# HackRx-specific route with Bearer token validation using ENV
@router.post("/hackrx/run", response_model=DocumentResponse)
async def run_hackrx_submission(
    request: DocumentRequest,
    authorization: str = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split("Bearer ")[-1]
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        ingest_document(request.documents)
        answers = answer_questions(
            document_url=request.documents,
            questions=request.questions,
            top_k=8
        )
        return DocumentResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
