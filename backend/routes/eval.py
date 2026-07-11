from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.eval_result import EvalResult
from backend.services.evaluator import evaluate_response
import uuid

router = APIRouter()

class EvalRequest(BaseModel):
    question:     str
    answer:       str
    contexts:     list[str]
    ground_truth: Optional[str] = ""
    model_used:   Optional[str] = ""

class EvalResponse(BaseModel):
    eval_id:           str
    faithfulness:      Optional[float]
    answer_relevancy:  Optional[float]
    context_precision: Optional[float]
    context_recall:    Optional[float]
    overall_score:     Optional[float]
    eval_duration_s:   float
    verdict:           str   # "PASS" / "REVIEW" / "FAIL"

class MetricsResponse(BaseModel):
    total_evaluations:      int
    avg_faithfulness:       Optional[float]
    avg_answer_relevancy:   Optional[float]
    avg_context_precision:  Optional[float]
    avg_overall_score:      Optional[float]
    recent_evals:           list


def get_verdict(overall_score: Optional[float]) -> str:
    if overall_score is None:
        return "FAIL"
    if overall_score >= 0.75:
        return "PASS"
    if overall_score >= 0.5:
        return "REVIEW"
    return "FAIL"


@router.post("/", response_model=EvalResponse)
def run_evaluation(req: EvalRequest, db: Session = Depends(get_db)):
    """Run RAGAS evaluation on a question-answer-context triplet."""

    # run evaluation
    scores = evaluate_response(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth or ""
    )

    # save to DB
    eval_id = str(uuid.uuid4())
    record = EvalResult(
        id=eval_id,
        question=req.question,
        answer=req.answer[:2000],
        faithfulness=scores.get("faithfulness"),
        answer_relevancy=scores.get("answer_relevancy"),
        context_precision=scores.get("context_precision"),
        context_recall=scores.get("context_recall"),
        overall_score=scores.get("overall_score"),
        eval_duration_s=scores.get("eval_duration_s", 0.0),
        model_used=req.model_used
    )
    db.add(record)
    db.commit()

    return EvalResponse(
        eval_id=eval_id,
        faithfulness=scores.get("faithfulness"),
        answer_relevancy=scores.get("answer_relevancy"),
        context_precision=scores.get("context_precision"),
        context_recall=scores.get("context_recall"),
        overall_score=scores.get("overall_score"),
        eval_duration_s=scores.get("eval_duration_s", 0.0),
        verdict=get_verdict(scores.get("overall_score"))
    )


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(db: Session = Depends(get_db)):
    """Get aggregated evaluation metrics across all stored evaluations."""

    records = db.query(EvalResult).order_by(EvalResult.created_at.desc()).all()

    if not records:
        return MetricsResponse(
            total_evaluations=0,
            avg_faithfulness=None,
            avg_answer_relevancy=None,
            avg_context_precision=None,
            avg_overall_score=None,
            recent_evals=[]
        )

    def safe_avg(values):
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    recent = []
    for r in records[:10]:
        recent.append({
            "id":                r.id[:8],
            "question":          r.question[:80] + "..." if len(r.question) > 80 else r.question,
            "overall_score":     r.overall_score,
            "faithfulness":      r.faithfulness,
            "answer_relevancy":  r.answer_relevancy,
            "context_precision": r.context_precision,
            "verdict":           get_verdict(r.overall_score),
            "created_at":        str(r.created_at)
        })

    return MetricsResponse(
        total_evaluations=len(records),
        avg_faithfulness=safe_avg([r.faithfulness for r in records]),
        avg_answer_relevancy=safe_avg([r.answer_relevancy for r in records]),
        avg_context_precision=safe_avg([r.context_precision for r in records]),
        avg_overall_score=safe_avg([r.overall_score for r in records]),
        recent_evals=recent
    )


@router.get("/history")
def get_eval_history(limit: int = 20, db: Session = Depends(get_db)):
    """Get full evaluation history for dashboard charts."""
    records = db.query(EvalResult).order_by(EvalResult.created_at.desc()).limit(limit).all()
    return [
        {
            "id":                r.id[:8],
            "question":          r.question[:60] + "..." if len(r.question) > 60 else r.question,
            "faithfulness":      r.faithfulness,
            "answer_relevancy":  r.answer_relevancy,
            "context_precision": r.context_precision,
            "overall_score":     r.overall_score,
            "verdict":           get_verdict(r.overall_score),
            "model_used":        r.model_used,
            "eval_duration_s":   r.eval_duration_s,
            "created_at":        str(r.created_at)
        }
        for r in records
    ]