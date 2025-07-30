from fastapi import APIRouter, HTTPException
from app.services.document_parser import extract_text
from app.services.query_parser import parse_query_with_hf as parse_query
from app.models.response_model import DocumentRequest, DocumentResponse

router = APIRouter()

@router.post("/process-document", response_model=DocumentResponse)
async def process_document(request: DocumentRequest):
    try:
        extracted_text = extract_text(request.documents)

        extracted_fields = []
        for q in request.questions:
            parsed = parse_query(q)
            extracted_fields.append(parsed)

        # ✅ Return a Pydantic model instance, not a dict
        return DocumentResponse(
            extracted_text=extracted_text,
            parsed_fields=extracted_fields
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
