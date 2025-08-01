from pydantic import BaseModel
from typing import List, Dict, Union

class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

class DocumentResponse(BaseModel):
    text: str  # Raw extracted text from the document
    context_chunks: List[str]  # Retrieved chunks from Pinecone
    prompt: str  # Prompt sent to the LLM
    llm_response: Union[Dict[str, Union[str, Dict]], List[Dict[str, Union[str, Dict]]]]  # Answer JSON from the model
