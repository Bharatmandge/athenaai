import time 
from groq import Groq
from backend.services.tavily_search import search_multiple
import os
from dotenv import load_dotenv
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SUMMARIZE_PROMPT = """Summarize the key information from these search results about: {query}

Results:
{results}

Write a concise, factual summary (3-5 sentences). Include specific facts and data points.
Do not mention that this came from search results."""

REPORT_PROMPT = """You are a research report writer.
Topic: {topic}

Research summaries from multiple sources:
{summaries}

Write a comprehensive, well-structured research report with:
1. An executive summary (2-3 sentences)
2. Key findings (bullet points)
3. Detailed analysis (3-4 paragraphs)
4. Conclusion

Use the summaries as your source material. Be factual and cite which aspect each point covers."""

def summarize_results(query: str, results: list[dict]) -> str:
    """Summarize search results for one sub-query."""
    if not results:
        return f"No results found for: {query}"

    results_text = "\n\n".join([
        f"Source: {r['title']} ({r['url']})\n{r['content']}"
        for r in results
    ])

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": SUMMARIZE_PROMPT.format(
                    query=query,
                    results=results_text[:4000]
                )
            }],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ResearchAgent] Summarize failed: {e}")
        return results_text[:500]  # return raw if summarize fails


def generate_report(topic: str, summaries: dict) -> str:
    """Generate final research report from all summaries."""
    summaries_text = "\n\n".join([
        f"### {query}\n{summary}"
        for query, summary in summaries.items()
    ])

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": REPORT_PROMPT.format(
                    topic=topic,
                    summaries=summaries_text[:6000]
                )
            }],
            temperature=0.4,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ResearchAgent] Report generation failed: {e}")
        return summaries_text

def run_research(topic: str, queries: list[str]) -> dict:
    """
    Full research Pipeline:
    1.  Search Tavily for each sub-query
    2.  Summarize each result set
    3.  Generate final report
    Returns dict with report, summaries, sources, citations
    """

    start = time.time()

    print(f"[ResearchAgent] Searching {len(queries)} queries....")
    all_results = search_multiple(queries, max_results_each=3)

    # Collect all source URL for citations 
    all_sources = []
    for query, results in all_results.items():
        for r in results:
            if r["url"] not in [s["url"] for s in all_sources]:
                all_sources.append({
                    "title":   r["title"],
                    "url":     r["url"],
                    "query":   query
                }) 
    # summarizes each query results 
    print(f"[ResearchAgent] Summarizing results....")
    summaries = {}
    for query, results in all_results.items():
        summaries[query] = summarize_results(query, results)

    # Generate final report 
    print(f"[ResearchAgent] Generating final report......")
    report = generate_report(topic, summaries)

    duration = round(time.time() - start, 2)
    print(f"[ResearchAgent] Done in {duration}s - {len(all_sources)} sources")

    return {
        "report": report, 
        "summaries": summaries,
        "sources": all_sources,
        "duration": duration
    }