from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

    avg_len = sum(len(c) for c in chunks) // len(chunks)
    print(f" Chunked into {len(chunks)} parts | Avg length: {avg_len} chars")
    return chunks