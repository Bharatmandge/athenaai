from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap = 200, 
    length_function=len, 
    separators=["\n\n", "\n", " ", ""]
)

def chunk_text(
    text: str, 
    doc_id: str, 
    filename: str
) -> List[Dict]:
    raw_chunks = splitter.split_text(text)
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{i}",
            "doc_id": doc_id, 
            "filename": filename, 
            "chunk_index": i, 
            "text": chunk_text, 
            "char_count": len(chunk_text),
        })
    return chunks