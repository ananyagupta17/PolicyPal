import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables!")

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)

system_prompt = """
You are an expert at extracting structured information from text.
You will be given a query and you must extract the following information:
- Age
- Gender
- Procedure
- City
- Policy duration

The output must be a single JSON object. If any information is not present in the text, use "N/A" for its value.
Example output:
{
  "age": "45",
  "gender": "male",
  "procedure": "appendectomy",
  "city": "New York City",
  "policy_duration": "3 years"
}
"""

def parse_query_with_hf(query: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=256,
            stream=False
        )

        full_response_text = response.choices[0].message.content.strip()

        # Try direct JSON parse first
        try:
            extracted_info = json.loads(full_response_text)
        except json.JSONDecodeError:
            # Try to extract from substring if raw response includes explanation before JSON
            json_start = full_response_text.find('{')
            json_end = full_response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                try:
                    json_str = full_response_text[json_start:json_end]
                    extracted_info = json.loads(json_str)
                except json.JSONDecodeError:
                    extracted_info = {
                        "error": "Invalid JSON format even after slicing.",
                        "raw_response": full_response_text
                    }
            else:
                extracted_info = {
                    "error": "No JSON object found in model output.",
                    "raw_response": full_response_text
                }

        print("🟢 Mistral Raw Response:\n", full_response_text)
        print("🟢 Parsed JSON:\n", extracted_info)

        return extracted_info

    except Exception as e:
        return {"error": str(e)}
