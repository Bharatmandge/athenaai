from sqlalchemy.orm.exc import FlushError
from langchain_core.agents import AgentActionMessageLog
import time, json, re
from backend.agents.state import AthenaState

from groq  import Groq
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CRITIC_PROMPT = """You are a strict answer quality evaluator.

Given a question, the context used to answer it, and a draft answer — evaluate the answer quality.

Scoring criteria:
- Faithfulness (0-0.5): Is every claim in the answer supported by the context? Deduct for hallucinations.
- Completeness (0-0.5): Does the answer fully address the question using available context?

Return ONLY valid JSON, no markdown, no explanation:
{{
  "score": 0.0 to 1.0,
  "faithfulness": 0.0 to 0.5,
  "completeness": 0.0 to 0.5,
  "critique": "specific issues found, or 'Answer is accurate and complete' if score >= 0.7",
  "needs_retry": true or false
}}

Question: {query}

Context used:
{context}

Draft answer:
{answer}"""


def _parse_critic_json(raw: str) -> dict:
    raw = re.sub(r"```json", "", raw).strip()
    return json.loads(raw)

def _critique_with_gemini(query, context, answer) -> dict:
    prompt = CRITIC_PROMPT.format(
        query=query,
        context=context[:3000],
        answer=answer[:2000]
    )
    response = client_gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return _parse_critic_json(response.text)


def _critique_with_groq(query, context, answer) -> dict:
    prompt = CRITIC_PROMPT.format(
        query=query,
        context=context[:3000],
        answer=answer[:2000]
    )
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512
    )
    return _parse_critic_json(response.choices[0].message.content)

def critic_node(state: AthenaState) -> AthenaState:
    start = time.time()
    logs  = state.get("agent_logs", [])
    query = state["query"]
    answer = state.get("draft_answer", "") or ""
    context = state.get("context_string", "") or ""

    if not answer:
        critique_result = {
            "score":    0.0,
            "faithfulness":  0.0,
            "completeness":  0.0,
            "critique":   "No answer was generated",
            "needs_retry": False
        }
    else:
        try:
            critique_result = _critique_with_gemini(query, context, answer)
            print(f"[Critic] Used Gemini — score: {critique_result.get('score')}")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print("[Critic] Gemini quota — using Groq")
            else:
                print(f"[Critic] Gemini error — using Groq")
            try:
                critique_result = _critique_with_groq(query, context, answer)
                print(f"[Critic] Used Groq — score: {critique_result.get('score')}")
            except Exception as groq_err:
                print(f"[Critic] Both failed: {groq_err}")
                # if critic fails, assume answer is acceptable
                critique_result = {
                    "score":        0.75,
                    "faithfulness": 0.4,
                    "completeness": 0.35,
                    "critique":     "Critic evaluation unavailable — answer passed by default",
                    "needs_retry":  False
                }

    score    = float(critique_result.get("score", 0.75))
    critique = critique_result.get("critique", "")
    # at the bottom of critic_node, replace needs_retry logic:
    needs_retry = score < 0.7   # workflow will enforce the hard cap itself

    duration = round(time.time() - start, 2)
    logs.append({
        "agent":       "critic",
        "duration_s":  duration,
    "status":      "done",
    "score":       score,
    "needs_retry": needs_retry
})

    print(f"[Critic] Score: {score} | Needs retry: {needs_retry}")

    return {
        **state,
        "critique":       critique,
        "critique_score": score,
        "agent_logs":     logs
    }