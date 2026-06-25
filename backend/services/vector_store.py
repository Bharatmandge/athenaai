from dotenv import load_dotenv
load_dotenv()

import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

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
            }
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION, points=points)