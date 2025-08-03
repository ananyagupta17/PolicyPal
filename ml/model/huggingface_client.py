import requests
import os
from typing import Union, List, Dict

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

def call_huggingface_llm(prompt: Union[str, List[Dict]]) -> str:
    # If prompt is a list of dicts with "text" fields (e.g. retrieval output), convert to string
    if isinstance(prompt, list):
        prompt = "\n".join([item.get("text", "") for item in prompt])

    payload = {"inputs": prompt}
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result[0]["generated_text"] if isinstance(result, list) else result
