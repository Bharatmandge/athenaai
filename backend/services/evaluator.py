import os, time, warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

# ── initialize ONCE at module level — never re-init inside functions ──────────

_groq_llm = LangchainLLMWrapper(
    ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
        # DO NOT pass n= — causes BadRequestError on newer langchain-groq
    )
)

# reuse cached bge-base-en-v1.5 — same model embedder.py already loaded
_embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder=os.path.expanduser("~/.cache/huggingface/hub")
)

_ragas_embeddings = LangchainEmbeddingsWrapper(_embed_model)

# sequential execution — max_workers=1 prevents n>1 BadRequestError
_run_config = RunConfig(
    max_workers=1,
    timeout=90,
    max_retries=1
)

# ─────────────────────────────────────────────────────────────────────────────


def _safe_float(val) -> float | None:
    """Convert value to float, return None if NaN or invalid."""
    try:
        f = float(val)
        # NaN check — NaN != NaN is always True
        return None if (f != f) else round(f, 4)
    except Exception:
        return None


def _empty_scores(error: str = "") -> dict:
    """Return empty score dict when evaluation cannot run."""
    return {
        "faithfulness":      None,
        "answer_relevancy":  None,
        "context_precision": None,
        "context_recall":    None,
        "overall_score":     None,
        "eval_duration_s":   0.0,
        "error":             error
    }


def _compute_answer_relevancy(question: str, answer: str) -> float | None:
    """
    Compute answer relevancy manually via embedding cosine similarity.
    Measures how semantically aligned the answer is with the question.
    No LLM call needed — no n>1 issue — fast and reliable.
    """
    try:
        from backend.services.embedder import model as embed_model

        q_vec = embed_model.encode(question, normalize_embeddings=True)
        a_vec = embed_model.encode(answer,   normalize_embeddings=True)

        # cosine similarity — vectors already normalized, so dot product = cosine
        score = float(np.dot(q_vec, a_vec))

        # clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        return round(score, 4)

    except Exception as e:
        print(f"[RAGAS] Answer relevancy computation failed: {e}")
        return None


def _compute_faithfulness(
    question: str,
    answer:   str,
    contexts: list[str]
) -> float | None:
    """
    Run RAGAS faithfulness metric only.
    Checks if every claim in the answer is supported by the retrieved contexts.
    Uses Groq LLM + sequential execution to avoid n>1 BadRequestError.
    """
    data = {
        "question": [question],
        "answer":   [answer],
        "contexts": [contexts],
    }

    try:
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=_groq_llm,
            embeddings=_ragas_embeddings,
            raise_exceptions=False,
            run_config=_run_config
        )

        df    = result.to_pandas()
        score = _safe_float(df.iloc[0].get("faithfulness"))
        return score

    except Exception as e:
        print(f"[RAGAS] Faithfulness computation failed: {e}")
        return None


def evaluate_response(
    question:     str,
    answer:       str,
    contexts:     list[str],
    ground_truth: str = ""
) -> dict:
    """
    Main evaluation function. Call this from routes/eval.py.

    Computes:
        faithfulness     — RAGAS metric via Groq LLM
                           checks if answer claims are supported by contexts
        answer_relevancy — manual embedding cosine similarity
                           checks if answer addresses the question

    Args:
        question:     the user's original query
        answer:       Athena's generated answer
        contexts:     list of retrieved chunk texts used to generate the answer
        ground_truth: optional reference answer (stored but not used in scoring)

    Returns:
        dict with faithfulness, answer_relevancy, overall_score, eval_duration_s
    """
    start = time.time()

    # ── input validation ──────────────────────────────────────────────────────
    if not contexts or all(not c.strip() for c in contexts):
        print("[RAGAS] ⚠️ No contexts provided — skipping evaluation")
        return _empty_scores("No contexts provided")

    if not answer or not answer.strip():
        print("[RAGAS] ⚠️ No answer provided — skipping evaluation")
        return _empty_scores("No answer provided")

    if not question or not question.strip():
        print("[RAGAS] ⚠️ No question provided — skipping evaluation")
        return _empty_scores("No question provided")

    # ── compute faithfulness via RAGAS ────────────────────────────────────────
    print("[RAGAS] Computing faithfulness...")
    faith = _compute_faithfulness(question, answer, contexts)

    # ── compute answer relevancy via embeddings ───────────────────────────────
    print("[RAGAS] Computing answer relevancy...")
    relev = _compute_answer_relevancy(question, answer)

    # ── compute overall score ─────────────────────────────────────────────────
    available = [v for v in [faith, relev] if v is not None]
    overall   = round(sum(available) / len(available), 4) if available else None

    duration = round(time.time() - start, 2)

    print(
        f"[RAGAS] ✅ Evaluation complete in {duration}s | "
        f"Faithfulness: {faith} | "
        f"Relevancy: {relev} | "
        f"Overall: {overall}"
    )

    return {
        "faithfulness":      faith,
        "answer_relevancy":  relev,
        "context_precision": None,   # excluded — requires reference + high token cost
        "context_recall":    None,   # excluded — requires reference + high token cost
        "overall_score":     overall,
        "eval_duration_s":   duration
    }