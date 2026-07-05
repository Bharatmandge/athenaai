import time
from backend.agents.state import AthenaState
from backend.graph.neo4j_client import run_query

def get_deep_graph_context(entity_names: list, depth: int = 3) -> tuple[str, list]:
    """
    Deep traversal - finds multi-hop paths between entities.
    Returns formatted context string + raw paths list.
    """
    if not entity_names:
        return "", []

    all_triples = []
    all_paths = []

    for entity in entity_names[:5]:
        # 1. get direct neighbors up to depth 3
        neighbors = run_query("""
            MATCH (e:Entity {name: $name})-[r*1..3]-(neighbor)
            RETURN DISTINCT
                neighbor.name AS neighbor_name,
                neighbor.type AS neighbor_type,
                type(r[-1])   AS relation
            LIMIT 20
        """, {"name": entity})

        for n in neighbors:
            if n.get("neighbor_name"):
                all_triples.append(
                    f"{entity} --[{n.get('relation', 'RELATED_TO')}]--> {n['neighbor_name']}"
                )
        
        # 2. Find shortest path between entity pairs if multiple entities
        if len(entity_names) > 1:
            for other in entity_names[1:3]:
                paths = run_query("""
                    MATCH path = shortestPath(
                        (a:Entity {name: $a})-[*..5]-(b:Entity {name: $b})
                    )
                    RETURN [n IN nodes(path) | n.name] AS path_nodes,
                           length(path) AS hops
                    LIMIT 3
                """, {"a": entity, "b": other})

                for p in paths:
                    if p.get("path_nodes"):
                        path_str = " -> ".join(p["path_nodes"])
                        all_paths.append({
                            "path":  path_str,
                            "hops": p.get("hops", 0),
                            "from":  entity,
                            "to":  other
                        })
                        all_triples.append(f"PATH:  {path_str}")

    seen = set()
    unique_triplets = []
    for t in all_triples:
        if t not in seen:
            seen.add(t)
            unique_triplets.append(t)

    context = "\n".join(unique_triplets[:40])
    return context, all_paths

def graph_agent_node(state: AthenaState) -> AthenaState:
    start  = time.time()
    logs   = state.get("agent_logs", [])
    entities = state.get("query_entities", [])

    try:
        deep_context, paths = get_deep_graph_context(entities, depth=3)
        status = "done"
        print(f"[GraphAgent] Found {len(deep_context.splitlines())} triples, {len(paths)} paths")
    except Exception as e:
        print(f"[GraphAgent] Failed: {e}")
        deep_context = ""
        paths        = []
        status       = "failed"

    duration = round(time.time() - start, 2)
    logs.append({
        "agent":      "graph_agent",
        "duration_s": duration,
        "status":     status,
        "triples_found": len(deep_context.splitlines()) if deep_context else 0
    })

    # merge deep graph context into existing graph context
    existing_graph = state.get("graph_context", "") or ""
    merged_graph   = (existing_graph + "\n" + deep_context).strip()

    # rebuild context string with enriched graph context
    from backend.retrieval.context_builder import build_context
    enriched_retrieval = {
        "vector_chunks": state.get("vector_chunks", []),
        "graph_context": merged_graph
    }
    enriched_context = build_context(enriched_retrieval)

    return {
        **state,
        "deep_graph_context": deep_context,
        "graph_paths":        paths,
        "graph_context":      merged_graph,
        "context_string":     enriched_context,
        "graph_context_used": bool(merged_graph),
        "agent_logs":         logs
    }