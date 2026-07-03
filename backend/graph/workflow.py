from langgraph.graph import StateGraph, END
from backend.agents.state import AthenaState
from backend.agents.planner_agent import planner_node
from backend.agents.retriever_agent import retriever_node
from backend.agents.responder_agent import responder_node
from backend.agents.fallback_agent import fallback_node

def should_use_fallback(state: AthenaState) -> str:
    """
    Conditional edge: if retriever found no chunks → fallback
    Otherwise → responder
    """
    chunks = state.get("vector_chunks", [])
    if not chunks:
        print("[Workflow] No chunks found - routing to fallback")
        return "fallback"

    print(f"[Workflow] {len(chunks)} chunks found - routing to responder")
    return "responder"


def build_workflow():
    workflow = StateGraph(AthenaState)

    workflow.add_node("planner",   planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("responder", responder_node)
    workflow.add_node("fallback",  fallback_node)

    # entry point
    workflow.set_entry_point("planner")

    # planner → retriever (always)
    workflow.add_edge("planner", "retriever")

    # retriever → responder OR fallback (conditional)
    workflow.add_conditional_edges(
        "retriever",
        should_use_fallback,
        {
            "responder": "responder",
            "fallback":  "fallback"
        }
    )

    # both responder and fallback → END
    workflow.add_edge("responder", END)
    workflow.add_edge("fallback",  END)

    return workflow.compile()

# compile once at module level — reused across all requests
athena_workflow = build_workflow()

