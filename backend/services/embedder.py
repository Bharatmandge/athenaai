from sentence_transformers import SentenceTransformer
import time
import os

# loaded ONCE at startup — shared with evaluator.py
model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    cache_folder=os.path.expanduser("~/.cache/huggingface/hub")
)

def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

def embed_batch(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    try:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )
        print(f"[Embedder] ✅ Embedded {len(texts)} chunks locally (bge-base-en-v1.5)")
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"[Embedder] ❌ Batch failed: {e}")
        return [[0.0] * 768 for _ in texts]