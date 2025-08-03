def build_llm_prompt(context_chunks, questions):
    """
    Builds a structured prompt for Gemini to answer questions based on document chunks.
    Ensures JSON-only response with no extra text.
    """
    context = "\n\n".join([chunk.get("text", "") for chunk in context_chunks])
    numbered_qs = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    return f"""
You are a helpful assistant. Use ONLY the excerpts below to answer the following questions.

If the answer is not clearly mentioned, respond with:
"I could not find this in the document."

Questions:
{numbered_qs}

Policy Excerpts (most relevant first):
{context}

Instructions:
- Answer concisely and accurately.
- Quote all figures/durations as-is.
- Do NOT make up information.
- Do NOT provide explanations or headings.
- Respond in the following strict JSON format:

{{
  "answers": [
    "Answer to question 1",
    "Answer to question 2",
    ...
  ]
}}

Return only this JSON. Nothing else.
""".strip()
