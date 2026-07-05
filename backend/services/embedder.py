from sentence_transformers import SentenceTransformer
import time

# loads once at startup, cached in memory — no API calls, no quota
_model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # 384-dim

# NOTE: your Qdrant collection is 768-dim
# we use "all-mpnet-base-v2" which produces 768-dim vectors
_model = SentenceTransformer("all-mpnet-base-v2")  # 768-dim ✓

def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    """Embed a single text string locally — no API, no quota."""
    embedding = _model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

def embed_batch(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    """Embed a list of texts locally in batch — fast and free."""
    try:
        embeddings = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        print(f"[Embedder] Embedded {len(texts)} chunks locally (all-mpnet-base-v2)")
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"[Embedder] Batch embed failed: {e}")
        return [[0.0] * 768 for _ in texts]