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

# NEW IMPORTS
from backend.graph.graph_builder import build_graph_for_chunk
import time

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
    filename = file.filename

    # --------------------------------------------------
    # 1. Parse document
    # --------------------------------------------------
    raw_text = parse_document(filename, file_bytes)

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
        filename
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
    # 5. Build Knowledge Graph
    # --------------------------------------------------
    for chunk in chunks:
        build_graph_for_chunk(
            chunk,
            doc_id,
            filename
        )
        time.sleep(0.5)  # Respect Gemini rate limits

    # --------------------------------------------------
    # 6. Update DB
    # --------------------------------------------------
    update_doc_status(doc_id, "graph_built")

    # --------------------------------------------------
    # Optional document record
    # --------------------------------------------------
    doc = Document(
        id=doc_id,
        filename=filename,
        filetype=ext,
        status="graph_built",
        chunk_count=len(chunks),
    )

    # TODO:
    # save_document(doc)

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "status": "graph_built"
    }