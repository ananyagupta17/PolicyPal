import hashlib
import json
from typing import List, Dict

from backend.services.retrieval import semantic_search  # your retrieval
from ml.model.huggingface_client import call_huggingface_llm  # LLM call
from ml.pipeline.prompt_builder import build_llm_prompt  # prompt builder
from backend.services.qa import answer_one_question  # heuristic fallback


def answer_questions(
    document_url: str,
    questions: List[str],
    top_k: int = 8,
) -> List[str]:
    """
    Full question-answer pipeline: retrieval + LLM + fallback.
    Returns a list of answers aligned with `questions`.
    """
    source_id = hashlib.md5(document_url.encode()).hexdigest()
    namespace = source_id  # you used namespace-per-document in upsert

    # 1. Retrieve shared context for all questions (optionally could do per-question)
    combined_query = " ".join(questions)
    context_chunks = semantic_search(
        combined_query,
        top_k=top_k,
        namespace=namespace,
        fltr={"source": {"$eq": source_id}},
    )

    # 2. Build LLM prompt with those chunks and the list of questions
    prompt = build_llm_prompt(context_chunks, questions)

    # 3. Call the LLM
    try:
        raw = call_huggingface_llm(prompt)
    except Exception as e:
        # LLM call failed: fallback to heuristics per question
        return [
            answer_one_question(
                q,
                semantic_search(
                    q,
                    top_k=top_k,
                    namespace=namespace,
                    fltr={"source": {"$eq": source_id}},
                ),
            )
            for q in questions
        ]

    # 4. Try to parse the LLM JSON output
    try:
        parsed = json.loads(raw.strip())
        answers = parsed.get("answers")
        if isinstance(answers, list) and len(answers) == len(questions):
            return answers
    except json.JSONDecodeError:
        pass  # will fall back below

    # 5. Fallback if JSON was invalid or misaligned
    fallback_answers = []
    for q in questions:
        retrieved = semantic_search(
            q,
            top_k=top_k,
            namespace=namespace,
            fltr={"source": {"$eq": source_id}},
        )
        fallback_answers.append(answer_one_question(q, retrieved))
    return fallback_answers
