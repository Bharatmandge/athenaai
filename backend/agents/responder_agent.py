import time 
from backend.agents.state import AthenaState
from backend.services.llm_service import generate_answer

def responder_node(state: AthenaState) -> AthenaState:
    start   = time.time()
    logs    = state.get("agent_logs", [])
    query   = state["query"]
    context = state.get("context_string", "")
    plan    = state.get("plan", "")
    citations = state.get("citations", [])

    # inject plan into query so LLM knows how to structure the answer
    enriched_query = f"""Answer plan:
{plan}

User question: {query}"""

    try:
        result = generate_answer(enriched_query, context, citations)
        answer     = result["answer"]
        model_used = result["model_used"]
        status     = "done"
    except Exception as e:
        print(f"[Responder] Failed: {e}")
        answer     = "I encountered an error generating the answer. Please try again."
        model_used = "none"
        status     = "failed"

    duration = round(time.time() - start, 2)
    logs.append({"agent": "responder", "duration_s": duration, "status": status})

    return {
        **state,
        "draft_answer": answer,
        "final_answer": answer,
        "model_used":   model_used,
        "agent_logs":   logs
    }
