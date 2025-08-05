def build_llm_prompt(context_chunks, questions):
    context = "\n\n".join([chunk.get("text", "") for chunk in context_chunks])
    numbered_qs = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    return f"""
You are a helpful and precise assistant tasked with answering questions strictly based on the provided policy excerpts below.

Your job is to write short, rephrased, **human-sounding one-line answers** for each question. Do **NOT** copy or splice raw text from the excerpts.

If the context does not clearly contain an answer, reply with:
"I could not find this information in the document."

⚠️ Instructions:
- ❌ Do NOT copy exact phrases or sentences from the excerpts.
- ❌ Do NOT include explanations, assumptions, or reasoning.
- ✅ DO paraphrase into clear, natural language.
- ✅ DO keep each answer to a **single sentence**.

🎯 Example format:

**Example Question:**
1. What is the waiting period for pre-existing conditions?

**Example Context:**
Pre-existing diseases shall be covered after a waiting period of 48 months from the date of commencement of the policy.

**Expected Answer Format (in JSON):**
{{
  "answers": [
    "Pre-existing conditions are covered after 48 months.",
    ...
  ]
}}

Now answer the following questions based ONLY on the context.

Questions:
{numbered_qs}

Policy Excerpts:
{context}
""".strip()
