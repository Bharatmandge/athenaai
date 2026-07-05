from typing import TypedDict, Optional

class AthenaState(TypedDict):
    query:      str
    plan:       Optional[str]

    vector_chunks:     Optional[list]
    graph_context:     Optional[str]
    query_entities:    Optional[list]
    context_string:    Optional[str]
    citations:         Optional[list]

    # graph agent  - NEW
    deep_graph_context:   Optional[str]
    graph_paths:          Optional[list]

    # responder 
    draft_answer:         Optional[str]
    critique_score:       Optional[float]
    retry_count:          Optional[int]

    # final 
    final_answer:         Optional[str]
    model_used:           Optional[str]
    graph_context_used:   Optional[bool]
    agent_logs:           Optional[list]

    
