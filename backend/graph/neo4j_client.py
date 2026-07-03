import os
from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
from dotenv import load_dotenv
load_dotenv()

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            max_connection_lifetime=200,   # close connections after 200s
            keep_alive=True
        )
    return _driver

def run_query(cypher: str, params: dict = {}) -> list[dict]:
    """Run a Cypher query with auto-reconnect on stale connection."""
    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(cypher, params)
            return [dict(r) for r in result]
    except (SessionExpired, ServiceUnavailable, OSError) as e:
        print(f"[Neo4j] Connection dropped, reconnecting... ({e})")
        # force new driver on next call
        global _driver
        _driver = None
        try:
            with get_driver().session() as session:
                result = session.run(cypher, params)
                return [dict(r) for r in result]
        except Exception as retry_err:
            print(f"[Neo4j] Retry failed: {retry_err}")
            return []   # return empty — never crash the pipeline
    except Exception as e:
        print(f"[Neo4j] Query error: {e}")
        return []

def verify_connection():
    result = run_query("RETURN 1 AS ok")
    if result:
        print("✅ Neo4j connected")
    else:
        print("⚠️ Neo4j connection failed — graph features disabled")