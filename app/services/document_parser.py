import fitz  # PyMuPDF
from docx import Document
import email
import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(file_path):
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting PDF text: {e}")
        raise

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    except Exception as e:
        logging.error(f"Error extracting DOCX text: {e}")
        raise

def extract_text_from_email(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            msg = email.message_from_file(f)

        parts = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    parts.append(part.get_payload(decode=True).decode(errors="ignore"))
                except:
                    parts.append(part.get_payload())

        return "\n".join(parts).strip()
    except Exception as e:
        logging.error(f"Error extracting email text: {e}")
        raise

#main router fucntion
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    logging.info(f"Extracting text from file: {file_path} (type: {ext})")

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".eml":
        return extract_text_from_email(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

#testing
if __name__ == "__main__":
    sample_path = "sample_email.eml"  
    result = extract_text(sample_path)
    print("\n--- Extracted Text ---\n")
    print(result)
