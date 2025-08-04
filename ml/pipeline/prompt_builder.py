def build_llm_prompt(context_chunks, questions):
    """
    Build a prompt that asks Gemini to answer questions clearly and concisely
    based ONLY on provided context chunks.
    """

    context = "\n\n".join([chunk.get("text", "") for chunk in context_chunks])
    numbered_qs = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    return f"""
You are a helpful, expert assistant answering questions strictly based on the provided policy excerpts below.

Please provide clear, concise, and natural-sounding one-line answers for each question. Use your understanding to summarize and rephrase the relevant information accurately — do NOT just copy-paste raw text fragments.

Answer using ONLY the information contained in the excerpts, which are ordered from most relevant to least relevant.

If the answer is not explicitly found in the excerpts, respond exactly with:
"I could not find this information in the document."

Do NOT add explanations or extra details beyond the answer.

Return your answers in the following JSON format ONLY (no extra text):

{{
  "answers": [
    "Answer to question 1",
    "Answer to question 2",
    ...
  ]
}}

Questions:
{numbered_qs}

Policy Excerpts:
{context}
""".strip()
