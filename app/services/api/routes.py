from fastapi import APIRouter, HTTPException
from app.services.document_parser import extract_text
from app.services.text_chunker import chunk_text
from app.services.embedding import get_embedding
from app.services.pinecone_store import upsert_chunks
from app.services.retrieval import semantic_search
from app.utils.prompt_builder import build_llm_prompt
from app.services.huggingface_client import call_huggingface_llm
from app.models.response_model import DocumentRequest, DocumentResponse

router = APIRouter()

@router.post("/process-document", response_model=DocumentResponse)
async def process_document(request: DocumentRequest):
    try:
        # 1. Extract raw text from uploaded documents
        extracted_text = extract_text(request.documents)

        # 2. Chunk the text
        chunks = chunk_text(extracted_text)

        # 3. Upsert to Pinecone (optional – only if this is the first upload)
        upsert_chunks(chunks, namespace=request.source_id)

        # 4. Semantic retrieval of relevant context chunks
        context_chunks = semantic_search(
            question=" ".join(request.questions),  # combined for broader match
            top_k=8,
            namespace=request.source_id
        )

        # 5. Build the LLM prompt
        prompt = build_llm_prompt(context_chunks, request.questions)

        # 6. Call HuggingFace LLM
        llm_response = call_huggingface_llm(prompt)

        # 7. Return everything
        return DocumentResponse(
            text=extracted_text,
            context_chunks=context_chunks,
            prompt=prompt,
            llm_response=llm_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
