from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

from backend.services.llm_service import generate_answer
from backend.retrieval.context_builder import build_citation_list, build_context
from backend.retrieval.hybrid_retriever import hybrid_retrieve as hybrid_retriever
# removed import: backend.schemas.chat does not exist, using local ChatRequest definition
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer:             str
    citations:          list[str]
    query_entities:     list[str]
    graph_context_used: bool
    model_used:         str          # ← add this

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    retrieval = hybrid_retriever(req.query, req.top_k)
    context   = build_context(retrieval)
    citations = build_citation_list(retrieval["vector_chunks"])
    result    = generate_answer(req.query, context, citations)

    return ChatResponse(
        answer=result["answer"],
        citations=citations,
        query_entities=retrieval["query_entities"],
        graph_context_used=bool(retrieval["graph_context"]),
        model_used=result["model_used"]    # corrected key
    )