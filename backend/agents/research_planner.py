import time , re, json
from groq import Groq
import os 
from dotenv import load_dotenv
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RESEARCH_PLANNER_PROMPT = """You are a research planning agent.
Given a research topic, break it down into 3-5 specific search queries.
Each query should target a different angle of the topic.
Return ONLY a JSON array of strings. No explanation, no markdown.

Topic: {topic}

Example output: ["query 1", "query 2", "query 3"]"""

def plan_research_queries(topic: str) -> list[str]:
    """Break research topic into focused sub-queries."""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": RESEARCH_PLANNER_PROMPT.format(topic=topic)
            }],
            temperature=0.3,
            max_tokens=512
        )
        raw = re.sub(r"```json|```", "", response.choices[0].message.content).strip()
        queries = json.loads(raw)
        print(f"[ResearchPlanner] Generated {len(queries)} sub-queries")
        return queries[:5]  # max 5
    except Exception as e:
        print(f"[ResearchPlanner] Failed: {e} — using topic as single query")
        return [topic]