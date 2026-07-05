from tavily import TavilyClient
import os 
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    searched the web for a query using Tavily.
    Returns list of {title, url, content, score}.
    """
    try:
        response = tavily.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False
        )
        results= []
        for r in response.get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", "")[:1500],  # cap per source
                "score":   round(r.get("score", 0), 4)
            })
        print(f"[Tavily] '{query}' -> {len(results)} results")
        return results
    except Exception as e:
        print(f"[Tavily] Search failed for '{query}': {e}")
        return []

def search_multiple(queries: list[str], max_results_each: int = 3) -> dict:
    """
    Run multiple searches and return results keyed by query.
    """
    all_results = {}
    for query in queries:
        all_results[query] = search_web(query, max_results=max_results_each)
    return all_results
