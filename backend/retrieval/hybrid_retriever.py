from concurrent.futures import ThreadPoolExecutor
from backend.services.vector_store import similarity_search 
from backend.graph.graph_retriever import get_graph_context_for_query
import google.generativeai as genai 
import os, json, re 

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-1.5-flash")

def extract_query_entities(query: str) -> list[str]:
    """Pull entity names from user query using Gemini."""
    prompt = f"""Extract entity names from this query.
Return ONLY a JSON array of strings. No explanation, no markdown.
Query: {query}
Example output: ["LangChain", "OpenAI", "RAG"]"""
    try:
        resp = _model.generate_content(prompt)
        raw = re.sub(r"```json|```", "", resp.text).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Entity extraction failed: {e}")
        return []

def hybrid_retriever(query: str, top_k: int = 5) -> dict:
    """Run Qdrant vector search + neo4j graph lookup in parallel"""
    entities = extract_query_entities(query)

    with ThreadPoolExecutor(max_workers=2) as ex:
        vec_future = ex.submit(similarity_search, query, top_k)
        graph_future = ex.submit(get_graph_context_for_query, entities)

    return {
        "vector_chunks": vec_future.result(),
        "graph_context": graph_future.result(),
        "query_entities": entities
    }