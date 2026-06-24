# pyrefly: ignore [missing-import]
import PyPDF2, io 
from docx import Document as DocxDocument

def parse_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or ""
             for page in reader.pages]
    return "\n\n".join(pages)

def parse_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paras)

def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def parse_document(filename: str, file_bytes: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf": return parse_pdf(file_bytes)
    if ext == "docx": return parse_docx(file_bytes)
    if ext == "txt": return parse_txt(file_bytes)
    raise ValueError(f"Unsupported file type: {ext}")