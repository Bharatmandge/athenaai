from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

from backend.services.embedder import embed_text
from backend.services.vector_store import (
    client,
    COLLECTION,
)

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_id: Optional[str] = None


def similarity_search(
    query: str,
    top_k: int = 5,
    doc_id: Optional[str] = None,
):

    query_vector = embed_text(
        query,
        task_type="retrieval_query"
    )

    search_filter = None

    if doc_id:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    ).points

    return [
        {
            "text": r.payload["text"],
            "score": round(r.score, 4),
            "filename": r.payload["filename"],
            "chunk_index": r.payload["chunk_index"],
        }
        for r in results
    ]


@router.post("/search")
async def search(request: SearchRequest):

    results = similarity_search(
        query=request.query,
        top_k=request.top_k,
        doc_id=request.doc_id,
    )

    return {
        "query": request.query,
        "results": results,
    }

@router.get("/count")
def get_collection_count():
    from backend.services.vector_store import client, COLLECTION
    try:
        info = client.get_collection(COLLECTION)
        return {
            "collection": COLLECTION,
            "total_points": info.points_count,
            "status": info.status
        }
    except Exception as e:
        return {"error": str(e)}