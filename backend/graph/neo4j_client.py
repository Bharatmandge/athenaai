import os
from dotenv import load_dotenv
load_dotenv()

_driver = None

def get_driver():
    global _driver
    uri = os.getenv("NEO4J_URI")
    if not uri:
        return None  # no URI configured — graph features disabled
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            uri,
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            max_connection_lifetime=200,
            keep_alive=False,           # prevents infinite retry spam when DB is unreachable
            connection_timeout=5,       # fail fast instead of hanging
        )
    return _driver

def run_query(cypher: str, params: dict = {}) -> list[dict]:
    """Run a Cypher query with auto-reconnect on stale connection."""
    from neo4j.exceptions import SessionExpired, ServiceUnavailable
    driver = get_driver()
    if driver is None:
        return []   # graph disabled — no URI set
    try:
        with driver.session() as session:
            result = session.run(cypher, params)
            return [dict(r) for r in result]
    except (SessionExpired, ServiceUnavailable, OSError) as e:
        print(f"[Neo4j] Connection dropped, reconnecting... ({e})")
        global _driver
        _driver = None
        try:
            d = get_driver()
            if d is None:
                return []
            with d.session() as session:
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