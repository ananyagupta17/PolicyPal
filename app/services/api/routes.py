from fastapi import APIRouter, HTTPException
from app.services.document_parser import extract_text
from app.models.response_model import DocumentRequest, DocumentResponse

router = APIRouter()

@router.post("/process-document", response_model=DocumentResponse)
async def process_document(request: DocumentRequest):
    try:
        extracted_text = extract_text(request.documents)
        return DocumentResponse(text=extracted_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
