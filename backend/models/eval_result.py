from sqlalchemy import Column, String, Float, DateTime, Text
from sqlalchemy.sql import func
from backend.database import Base
import uuid

class EvalResult(Base):
    __tablename__ = "eval_results"
    id                = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question          = Column(Text,   nullable=False)
    answer            = Column(Text,   nullable=False)
    faithfulness      = Column(Float,  nullable=True)
    answer_relevancy  = Column(Float,  nullable=True)
    context_precision = Column(Float,  nullable=True)
    context_recall    = Column(Float,  nullable=True)
    overall_score     = Column(Float,  nullable=True)
    eval_duration_s   = Column(Float,  nullable=True)
    model_used        = Column(String, nullable=True)
    created_at        = Column(DateTime, server_default=func.now())
