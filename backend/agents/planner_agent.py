
import time 
from backend.agents.state import AthenaState
from groq import Groq

import os, re
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PLANNER_PROMPT = """You are a query planning agent.
Given a user question, generate a brief 3-step plan for how to answer it.
Be concise. Return plain text, no JSON, no markdown.

Question: {query}

Plan (3 steps max):"""

def _plan_with_gemini(query: str) -> str:
    response = client_gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=PLANNER_PROMPT.format(query=query)
    )
    return response.text.strip()

def _plan_with_groq(query: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": PLANNER_PROMPT.format(query=query)}],
        temperature=0.3,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

def planner_node(state: AthenaState) -> AthenaState:
    start = time.time()
    logs = state.get("agent_logs", [])
    query = state["query"]

    try:
        plan = _plan_with_gemini(query)
        print(f"[Planner] Used Gemini")
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            print(f"[Planner] Gemini quota — using Groq")
        else:
            print(f"[Planner] Gemini error — using Groq")
        try:
            plan = _plan_with_groq(query)
            print(f"[Planner] Used Groq")
        except Exception as groq_err:
            plan = f"1. Search documents\n2. Find relevant info\n3. Answer the question"
            print(f"[Planner] Both failed, using default plan")
    
    duration = round(time.time() - start, 2)
    logs.append({"agent": "planner", "duration_s": duration, "status": "done"})
    
    return {
        **state, 
        "plan": plan,  
        "agent_logs": logs
    }