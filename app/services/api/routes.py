from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel

from app.services.pipeline_qa import answer_questions
from app.services.pinecone_store import ingest_document

router = APIRouter()


class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]


class DocumentResponse(BaseModel):
    answers: List[str]


@router.post("/process-document", response_model=DocumentResponse)
async def process_document(request: DocumentRequest):
    try:
        # 1. Ingest the document (extract, chunk, embed)
        ingest_document(request.documents)

        # 2. Use the QA pipeline to get answers
        answers = answer_questions(
            document_url=request.documents,
            questions=request.questions,
            top_k=8
        )

        # 3. Return only the answers
        return DocumentResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
