from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.graph.workflow import athena_workflow

router = APIRouter()

class ChatRequest(BaseModel):
    query:  str
    doc_id: Optional[str] = None
    top_k:  int = 5

class ChatResponse(BaseModel):
    answer:             str
    citations:          list
    query_entities:     list
    graph_context_used: bool
    model_used:         str
    agent_logs:         list
    plan:               str

@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    initial_state = {
        "query":              req.query,
        "plan":               None,
        "vector_chunks":      None,
        "graph_context":      None,
        "query_entities":     None,
        "context_string":     None,
        "draft_answer":       None,
        "final_answer":       None,
        "citations":          None,
        "model_used":         None,
        "agent_logs":         [],
        "graph_context_used": False
    }

    result = athena_workflow.invoke(initial_state)

    return ChatResponse(
        answer=result.get("final_answer") or "No answer generated",
        citations=result.get("citations") or [],
        query_entities=result.get("query_entities") or [],
        graph_context_used=result.get("graph_context_used") or False,
        model_used=result.get("model_used") or "unknown",
        agent_logs=result.get("agent_logs") or [],
        plan=result.get("plan") or ""
    )