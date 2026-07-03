from backend.graph.neo4j_client import run_query

def get_entity_neighbors(entity_name: str, depth: int = 2) -> list:
    safe_depth = int(depth)
    return run_query(f"""
        MATCH (e:Entity {{name: $name}})-[r*1..{safe_depth}]-(neighbor)
        RETURN DISTINCT neighbor.name AS name,
               labels(neighbor)[0] as label, 
               type(last(r)) AS relation
        LIMIT 30           
    """, {"name": entity_name})

def get_document_entities(doc_id: str) -> list:
    return run_query("""
        MATCH (d:Document {doc_id: $doc_id})-[:MENTIONS]->(e:Entity)
        RETURN DISTINCT e.name AS name, e.type AS type
    """, {"doc_id": doc_id})

def get_graph_context_for_query(entity_names: list[str]) -> str:
    if not entity_names:
        return ""
    results = []
    for name in entity_names[:5]:
        neighbors = get_entity_neighbors(name, depth=2)
        for n in neighbors:
            results.append(f"{name} --[{n['relation']}]--> {n['name']}")
    return "\n".join(results)

def find_path_between(entity_a: str, entity_b: str) -> list:
    return run_query("""
        MATCH path = shortestPath(
            (a:Entity {name: $a})-[*..5]-(b:Entity {name: $b})
        )
        RETURN [n IN nodes(path) | n.name] AS path_nodes,
               length(path) AS hops
    """, {"a": entity_a, "b": entity_b})

def get_graph_context_for_query(entity_names: list) -> str:
    if not entity_names:
        return ""
    try:
        results = []
        for name in entity_names[:5]:
            neighbors = get_entity_neighbors(name, depth=2)
            for n in neighbors:
                results.append(f"{name} --[{n.get('relation', 'RELATED_TO')}]--> {n.get('name', '')}")
        return "\n".join(results)
    except Exception as e:
        print(f"[GraphRetriever] Failed: {e} — continuing without graph context")
        return ""   # chat still works, just without graph enrichment