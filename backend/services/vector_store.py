from backend.services.embedder import embed_batch
from dotenv import load_dotenv
load_dotenv()


import os
import uuid

from qdrant_client import QdrantClient
from backend.services.embedder import embed_text

from qdrant_client.http.models import VectorParams, PointStruct, Distance


print("QDRANT_URL =", os.getenv("QDRANT_URL"))

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    check_compatibility=False,
)

COLLECTION = "athena_chunks"

def create_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    # create payload index for doc_id filtering
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="doc_id",
        field_schema="keyword"
    )

def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "text": chunks[i]["text"],
                "doc_id": chunks[i]["doc_id"],
                "filename": chunks[i]["filename"],
                "chunk_index": chunks[i]["chunk_index"],
            },
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION, points=points)


def similarity_search(query: str, top_k: int = 5) -> list[dict]:
    from backend.services.embedder import embed_text

    # task_type param kept for compatibility but local model ignores it
    query_vec = embed_text(query, task_type="retrieval_query")

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=top_k,
        with_payload=True
    ).points

    return [
        {
            "text":        r.payload["text"],
            "score":       round(r.score, 4),
            "filename":    r.payload["filename"],
            "chunk_index": r.payload["chunk_index"],
        }
        for r in results
    ]