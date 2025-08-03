import os
import google.generativeai as genai
from typing import Union

# Load Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

def call_gemini_llm(prompt: str) -> str:
    """
    Sends a prompt string to Gemini and returns the plain text response.
    Assumes prompt is already fully formatted.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Optional: You could raise the error if you want the pipeline to fail loudly
        return f"Error calling Gemini API: {str(e)}"
