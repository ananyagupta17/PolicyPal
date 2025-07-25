# app/services/document_parser.py

import requests
import mimetypes
import fitz  # PyMuPDF
import docx

def download_file(url, filename="temp_file"):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download document: {response.status_code}")
    
    content_type = response.headers.get('content-type')
    ext = mimetypes.guess_extension(content_type) or ".pdf"
    filepath = f"{filename}{ext}"

    with open(filepath, "wb") as f:
        f.write(response.content)
    
    return filepath

def extract_text_from_pdf(filepath):
    text = ""
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_document(url):
    filepath = download_file(url)

    if filepath.endswith(".pdf"):
        return extract_text_from_pdf(filepath)
    elif filepath.endswith(".docx"):
        return extract_text_from_docx(filepath)
    else:
        raise Exception("Unsupported document type")
