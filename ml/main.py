from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from ml.pipeline.pipeline_qa import answer_questions

app = FastAPI(
    title="ML QA Microservice",
    description="Handles prompt building, LLM calls, and fallback heuristics for QA.",
    version="1.0.0"
)

class QARequest(BaseModel):
    document_url: str
    questions: List[str]
    top_k: int = 8

class QAResponse(BaseModel):
    answers: List[str]

@app.post("/answer-questions", response_model=QAResponse)
def get_answers(payload: QARequest):
    try:
        answers = answer_questions(
            document_url=payload.document_url,
            questions=payload.questions,
            top_k=payload.top_k
        )
        return QAResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
