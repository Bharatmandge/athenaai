from sentence_transformers import SentenceTransformer
import time

# 120MB, 768-dim, fast, production-ready
_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    """Embed single text locally — no API, no quota, no rate limits."""
    embedding = _model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embedding.tolist()

def embed_batch(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    """Embed list of texts in one batch — fast local inference."""
    try:
        embeddings = _model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )
        print(f"[Embedder]  Embedded {len(texts)} chunks locally (bge-base-en-v1.5)")
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"[Embedder]  Batch failed: {e}")
        return [[0.0] * 768 for _ in texts]