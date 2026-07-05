from backend.agents.critic_agent import critic_node
from langgraph.graph import StateGraph, END
from backend.agents.state import AthenaState
from backend.agents.planner_agent import planner_node
from backend.agents.retriever_agent import retriever_node
from backend.agents.responder_agent import responder_node
from backend.agents.fallback_agent import fallback_node
from backend.agents.graph_agent import graph_agent_node

def route_after_retriever(state: AthenaState) -> str:
    """After retriever: if no chunks → fallback, else → graph_agent."""
    chunks = state.get("vector_chunks", [])
    if not chunks:
        print("[Workflow] No chunks — routing to fallback")
        return "fallback"
    print(f"[Workflow] {len(chunks)} chunks — routing to graph_agent")
    return "graph_agent"


def route_after_critic(state: AthenaState) -> str:
    score       = state.get("critique_score", 1.0) or 1.0
    retry_count = state.get("retry_count", 0) or 0

    print(f"[Workflow] route_after_critic — score: {score}, retry_count: {retry_count}")

    # HARD CAP — if retry already happened once, force END no matter what
    if retry_count >= 1:
        print(f"[Workflow] Hard cap hit (retry_count={retry_count}) — forcing END")
        return "end"

    if score < 0.7:
        print(f"[Workflow] Score {score} < 0.7 — one retry allowed")
        return "responder"

    print(f"[Workflow] Score {score} accepted — routing to END")
    return "end"


# ---Build workflow --------

def build_workflow():
    workflow = StateGraph(AthenaState)


    # Registering all nodes 
    workflow.add_node("planner",    planner_node)
    workflow.add_node("retriever",   retriever_node)
    workflow.add_node("graph_agent", graph_agent_node)
    workflow.add_node("responder",   responder_node)
    workflow.add_node("critic",   critic_node)
    workflow.add_node("fallback",   fallback_node)


    # entry point 
    workflow.set_entry_point("planner")
    
    # planner 
    workflow.add_edge("planner",  "retriever")

    # Retriever 
    workflow.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "graph_agent": "graph_agent",
            "fallback":   "fallback"
        }
    )

    # graph agent 
    workflow.add_edge("graph_agent", "responder")

    # responder 
    workflow.add_edge("responder", "critic")

    # critic -> responder 
    workflow.add_conditional_edges(
        "critic", 
        route_after_critic,
        {
            "responder": "responder",
            "end":   END
        }
    )
    
    # Fallback -> END
    workflow.add_edge("fallback", END)
    return workflow.compile()

athena_workflow = build_workflow()

