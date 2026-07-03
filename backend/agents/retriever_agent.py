import time 
from backend.agents.state import AthenaState
from backend.retrieval.hybrid_retriever import hybrid_retrieve
from backend.retrieval.context_builder import build_context, build_citation_list

def retriever_node(state: AthenaState) -> AthenaState:
    start = time.time()
    logs = state.get("agent_logs", [])
    query = state["query"]

    try:
        retrieval      = hybrid_retrieve(query, top_k=5)
        context_string = build_context(retrieval)
        citations      = build_citation_list(retrieval["vector_chunks"])

        duration = round(time.time() - start, 2)
        logs.append({
            "agent":      "retriever",
            "duration_s": duration,
            "status":     "done",
            "chunks_found": len(retrieval["vector_chunks"]),
            "graph_used":   bool(retrieval["graph_context"])
        })

        return {
            **state,
            "vector_chunks":     retrieval["vector_chunks"],
            "graph_context":     retrieval["graph_context"],
            "query_entities":    retrieval["query_entities"],
            "context_string":    context_string,
            "citations":         citations,
            "graph_context_used": bool(retrieval["graph_context"]),
            "agent_logs":        logs
        }

    except Exception as e:
        print(f"[Retriever] Failed: {e}")
        duration = round(time.time() - start, 2)
        logs.append({"agent": "retriever", "duration_s": duration, "status": "failed"})
        return {
            **state,
            "vector_chunks":  [],
            "graph_context":  "",
            "query_entities": [],
            "context_string": "",
            "citations":      [],
            "graph_context_used": False,
            "agent_logs":     logs
        }