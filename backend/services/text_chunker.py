from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 250) -> list[str]:
    """
    Split text into overlapping chunks suitable for embedding and retrieval.
    Defaults are tuned for policy/contract prose.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(text)
    # Trim empties / whitespace-only chunks
    return [c.strip() for c in chunks if c.strip()]
