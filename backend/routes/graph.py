from fastapi import APIRouter
from backend.graph.graph_retriever import get_entity_neighbors

router = APIRouter()

@router.get("/{node_name}")
def get_graph_neighbors(node_name: str, depth: int = 2):
    neighbors = get_entity_neighbors(node_name, depth)
    nodes = [{"id": node_name, "data": {"label": node_name}}]
    edges = []
    seen = set()
    for n in neighbors:
        nid = n["name"]
        if nid not in seen:
            nodes.append({"id": nid, "data": {"label": nid, "type": n.get("label")}})
            seen.add(nid)
        edges.append({
            "id": f"e-{node_name}-{nid}",
            "source": node_name,
            "target": nid, 
            "label": n.get("relation", "")
        })
    return {"nodes": nodes, "edges": edges}

    