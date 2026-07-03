from typing import TypedDict, Optional

class AthenaState(TypedDict):
    query:      str
    plan:       Optional[str]

    vector_chunks:     Optional[list]
    graph_context:     Optional[str]
    query_entities:    Optional[list]
    context_string:    Optional[str]

    draft_answer:      Optional[str]
    citations:         Optional[list]
    final_answer:      Optional[str]
    model_used:        Optional[str]

    agent_logs:        Optional[list]
    graph_context_used:   Optional[bool]

    