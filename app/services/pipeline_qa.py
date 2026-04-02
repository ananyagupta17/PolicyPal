import os
import hashlib
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

from app.services.retrieval import semantic_search
from app.utils.prompt_builder import build_chat_prompt

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# We use gemini-1.5-flash — fast, free tier, great for QA
model = genai.GenerativeModel("gemini-1.5-flash")


def answer_question(
    document_url: str,
    question: str,
    chat_history: List[Dict],
    top_k: int = 8,
    session_id: str = None,
) -> str:
    """
    Given a document URL, a user question, and the conversation
    history so far, retrieve relevant chunks and ask Gemini to answer.

    Chat history gives Gemini memory — it can handle follow-up questions
    like "what about for senior citizens?" without losing context.
    """
    # Use session_id directly if provided, otherwise compute from URL
    source_id = session_id if session_id else hashlib.md5(document_url.encode()).hexdigest()

    # 2. Retrieve the most relevant chunks for this question
    context_chunks = semantic_search(
        question,
        top_k=top_k,
        namespace=source_id,
        fltr={"source": {"$eq": source_id}},
    )

    # 3. Build the prompt — context + chat history + new question
    prompt = build_chat_prompt(
        context_chunks=context_chunks,
        chat_history=chat_history,
        question=question,
    )

    # 4. Call Gemini and get the answer
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Sorry, I couldn't generate an answer. Error: {str(e)}"