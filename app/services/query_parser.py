import os
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key  # Set the API key for openai client

def parse_query(user_query: str):
    prompt = f"""
Extract the following information from this query:
- Age
- Gender
- Procedure
- City
- Policy duration

Query: "{user_query}"

Return the result in this format:
{{
    "age": ...,
    "gender": "...",
    "procedure": "...",
    "city": "...",
    "policy_duration": "..."
}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']
