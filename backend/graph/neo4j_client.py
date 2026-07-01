import os 
from neo4j import GraphDatabase 
from dotenv import load_dotenv
load_dotenv()

_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

def run_query(cypher: str, params: dict = {}) -> list[dict]:
    with _driver.session() as session:
        result = session.run(cypher, params)
        return [dict(r) for r in result]

def verify_connections():
    try:
        _driver.verify_connectivity()
        print(" Neo4j Connected!")
    except Exception as e:
        print("wrong", e)