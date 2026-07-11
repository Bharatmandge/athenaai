from backend.graph.entity_extractor import extract_entities
from backend.graph.neo4j_client import run_query
import time 

def create_document_node(doc_id: str, filename: str):
    run_query("""
        MERGE (d:Document {doc_id: $doc_id})
        SET d.filename = $filename
    """, {
        "doc_id": doc_id,
        "filename": filename
    })


def create_entity_nodes(entities: list, doc_id: str):
    for ent in entities:
        run_query("""
            MERGE (e:Entity {name: $name})
            SET e.type = $type

            WITH e
            MATCH (d:Document {doc_id: $doc_id})

            MERGE (d)-[:MENTIONS]->(e)
        """, {
            "name": ent["name"],
            "type": ent["type"],
            "doc_id": doc_id
        })


def create_relationship(relationships: list):
    for rel in relationships:
        relation = rel["relation"].upper().replace(" ", "_")

        cypher = f"""
            MERGE (a:Entity {{name: $source}})
            MERGE (b:Entity {{name: $target}})
            MERGE (a)-[:{relation}]->(b)
        """

        run_query(cypher, {
            "source": rel["source"],
            "target": rel["target"]
        })


def build_graph_for_chunk(chunk: dict, doc_id: str, filename: str):
    extracted = extract_entities(chunk["text"])

    create_document_node(doc_id, filename)
    create_entity_nodes(extracted.get("entities", []), doc_id)
    create_relationship(extracted.get("relationships", []))
    time.sleep(1.5)