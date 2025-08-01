from fastapi import APIRouter, HTTPException
from app.models.response_model import DocumentRequest, DocumentResponse
from app.services.pipeline_qa import answer_questions
from app.services.pinecone_store import ingest_document

router = APIRouter()

@router.post("/process-document", response_model=DocumentResponse)
async def process_document(request: DocumentRequest):
    try:
        # 1. Ingest the document (extract, chunk, embed)
        source_id = ingest_document(request.documents)

        # 2. Use the QA pipeline to get answers
        answers = answer_questions(
            document_url=request.documents,
            questions=request.questions,
            top_k=8
        )

        # 3. Return the answers (and optionally the source_id)
        return DocumentResponse(
            text=source_id,  # or change to extracted_text if needed
            answers=answers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
