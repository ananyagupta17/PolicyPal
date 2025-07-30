from pydantic import BaseModel
from typing import List, Dict, Union

class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

class DocumentResponse(BaseModel):
    extracted_text: str
    parsed_fields: List[Dict[str, Union[str, None]]]


