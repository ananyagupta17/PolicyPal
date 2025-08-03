def build_llm_prompt(context_chunks, questions):
    context = "\n\n".join([chunk.get("text", "") for chunk in context_chunks])
    numbered_qs = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    return f"""
You answer strictly from the provided policy excerpts.
If the document does not specify, say: "I could not find this in the document."

Questions:
{numbered_qs}

Policy excerpts (most relevant first):
{context}

Instructions:
- Be brief and precise.
- Quote figures/durations exactly as written (e.g., "two (2) years", "15% of SI").
- Do not invent information not present in the excerpts.

Respond only in the following JSON format:

{{
  "answers": [
    "Answer to question 1",
    "Answer to question 2",
    ...
  ]
}}

Do not include any text or explanation outside the JSON.
""".strip()
