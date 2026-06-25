from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.document_parser import parse_document
from backend.services.chunker import chunk_text
from backend.services.embedder import embed_batch
from backend.services.vector_store import (
    create_collection,
    upsert_chunks
)

from backend.database import update_doc_status
from backend.models.document import Document

import uuid

router = APIRouter()

ALLOWED = {"pdf", "docx", "txt"}





@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()

    if ext not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Only pdf, docx and txt allowed. You got: {ext}"
        )

    file_bytes = await file.read()
    doc_id = str(uuid.uuid4())

    # --------------------------------------------------
    # 1. Parse document
    # --------------------------------------------------
    raw_text = parse_document(file.filename, file_bytes)

    if not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from this file."
        )

    # --------------------------------------------------
    # 2. Chunk document
    # --------------------------------------------------
    chunks = chunk_text(
        raw_text,
        doc_id,
        file.filename
    )

    # --------------------------------------------------
    # 3. Generate embeddings
    # --------------------------------------------------
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_batch(texts)

    # --------------------------------------------------
    # 4. Store vectors
    # --------------------------------------------------
    upsert_chunks(chunks, embeddings)

    # --------------------------------------------------
    # 5. Update DB
    # --------------------------------------------------
    update_doc_status(doc_id, "indexed")

    # --------------------------------------------------
    # Optional document record
    # --------------------------------------------------
    doc = Document(
        id=doc_id,
        filename=file.filename,
        filetype=ext,
        status="indexed",
        chunk_count=len(chunks),
    )

    # TODO:
    # save_document(doc)

    return {
        "doc_id": doc_id,
        "chunks_created": len(chunks),
        "status": "indexed"
    }