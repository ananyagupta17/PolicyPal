import hashlib
import json
from typing import List

from backend.services.retrieval import semantic_search  # your retrieval
from ml.model.gemini_client import call_gemini_llm  # ✅ changed to Gemini
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

    # 1. Retrieve shared context for all questions (shared chunks)
    combined_query = " ".join(questions)
    context_chunks = semantic_search(
        combined_query,
        top_k=top_k,
        namespace=namespace,
        fltr={"source": {"$eq": source_id}},
    )

    # 2. Build prompt for Gemini
    prompt = build_llm_prompt(context_chunks, questions)

    # 3. Call Gemini LLM
    try:
        raw_response = call_gemini_llm(prompt)
    except Exception:
        # Gemini call failed: fallback to heuristics
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

    # 4. Attempt to parse the JSON from Gemini
    try:
        # In case Gemini wraps response inside 'candidates[0].content.parts[0].text' format
        if isinstance(raw_response, dict):
            text = (
                raw_response.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
        else:
            text = raw_response

        parsed = json.loads(text.strip())
        answers = parsed.get("answers")

        if isinstance(answers, list) and len(answers) == len(questions):
            return answers
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass  # fallback

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
