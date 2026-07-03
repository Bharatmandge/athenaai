import time 
from backend.agents.state import AthenaState

def fallback_node(state: AthenaState) -> AthenaState:
    start = time.time()
    logs = state.get("agent_logs",  [])

    answer = (
        "I could not find relevant information in the uploaded documents "
        "to answer your question. Please upload documents related to this "
        "topic or rephrase your question."
    )

    duration = round(time.time() - start, 2)
    logs.append({"agent": "fallback", "duration_s": duration, "status": "triggered"})
    

    return {
        **state,
        "final_answer": answer, 
        "model_used": "none",
        "citations": [],
        "agent_logs":logs
    }