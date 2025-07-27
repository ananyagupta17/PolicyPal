from pydantic import BaseModel
from typing import List

class DocumentRequest(BaseModel):
    documents: str
    questions: List[str]

class DocumentResponse(BaseModel):
    text: str  # or whatever Step 2 returns
