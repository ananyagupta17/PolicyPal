from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
from pydantic import BaseModel
import os

from dotenv import load_dotenv
load_dotenv()

from ml.pipeline.pipeline_qa import answer_questions
from backend.services.pinecone_store import ingest_document

router = APIRouter()
security = HTTPBearer()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

class DocumentResponse(BaseModel):
    answers: List[str]

@router.post("/process-document", response_model=DocumentResponse)
async def process_document(request: DocumentRequest):
    try:
        print("📥 /process-document called")
        ingest_document(request.documents)
        print("📄 Document ingested")

        answers = answer_questions(
            document_url=request.documents,
            questions=request.questions,
            top_k=8
        )
        print("✅ Answers:", answers)

        return DocumentResponse(answers=answers)
    except Exception as e:
        print("🔥 Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hackrx/run", response_model=DocumentResponse)
async def run_hackrx_submission(
    request: DocumentRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    print("🚀 /hackrx/run endpoint hit")
    token = credentials.credentials
    print("🔐 Token received:", token)
    print("🔐 Expected token from env:", BEARER_TOKEN)

    if token != BEARER_TOKEN:
        print("❌ Token mismatch")
        raise HTTPException(status_code=403, detail="Invalid token")

    try:
        print("📥 Processing document:", request.documents)
        ingest_document(request.documents)
        print("📄 Document ingested")

        print("❓ Answering questions:", request.questions)
        answers = answer_questions(
            document_url=request.documents,
            questions=request.questions,
            top_k=8
        )
        print("✅ Final answers:", answers)

        return DocumentResponse(answers=answers)

    except Exception as e:
        print("🔥 ERROR in /hackrx/run:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
