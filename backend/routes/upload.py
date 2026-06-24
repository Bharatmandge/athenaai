from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.document_parser import parse_document
from backend.services.chunker import chunk_text
from backend.models.document import Document
import uuid 

router = APIRouter()
ALLOWED = {"pdf", "docx", "txt"}

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED:
        raise HTTPException(400, 
            detail=f"Only pdf, docx andtxt allowed you got :{ext}")

    file_bytes = await file.read()
    doc_id = str(uuid.uuid4())

    # 1. Parse 
    raw_text = parse_document(file.filename, file_bytes)
    if not raw_text.strip():
        raise HTTPException(422, detail="No text could be extracted from this file.")

    # chunks  
    chunks = chunk_text(raw_text, doc_id, file.filename)

    # store doc record 
    doc = Document(
        id=doc_id,
        filename=file.filename,
        filetype=ext,
        status="ready",
        chunk_count=len(chunks)
    )

    return {
        "doc_id":      doc_id,
        "filename":    file.filename,
        "chunk_count": len(chunks),
        "status":      "ready",
        "message":     "Document processed successfully"
    }
