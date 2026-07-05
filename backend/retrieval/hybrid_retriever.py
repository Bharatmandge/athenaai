from concurrent.futures import ThreadPoolExecutor
from backend.services.vector_store import similarity_search
from backend.graph.graph_retriever import get_graph_context_for_query

from groq import Groq
import os, json, re
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_query_entities(query: str) -> list[str]:
    prompt = f"""Extract entity names from this query.
Return ONLY a JSON array of strings. No explanation, no markdown.
Query: {query}
Example output: ["LangChain", "OpenAI", "RAG"]"""
    try:
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = re.sub(r"```json|```", "", response.text).strip()
        return json.loads(raw)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            print("[Query Entities] Gemini quota — trying Groq")
        else:
            print(f"[Query Entities] Gemini error — trying Groq")
    # fallback to Groq
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256
        )
        raw = re.sub(r"```json|```", "", response.choices[0].message.content).strip()
        return json.loads(raw)
    except Exception as groq_err:
        print(f"[Query Entities] Groq also failed: {groq_err}")
        return []   # empty list — pipeline continues without entity enrichment


def hybrid_retrieve(query: str, top_k: int = 5) -> dict:
    """Run Qdrant vector search + Neo4j graph lookup in parallel."""
    entities = extract_query_entities(query)

    with ThreadPoolExecutor(max_workers=2) as ex:
        vec_future   = ex.submit(similarity_search, query, top_k)
        graph_future = ex.submit(get_graph_context_for_query, entities)

    return {
        "vector_chunks":  vec_future.result(),
        "graph_context":  graph_future.result(),
        "query_entities": entities
    }