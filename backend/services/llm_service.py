# pyrefly: ignore [missing-import]
import google.generativeai as genai
# pyrefly: ignore [missing-import]
from groq import Groq
import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# configure clients
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are Athena, an intelligent knowledge assistant.
You have access to two types of context:
1. KNOWLEDGE GRAPH RELATIONSHIPS — structured triples showing how concepts connect
2. RELEVANT DOCUMENT CHUNKS — text extracted from uploaded documents

Rules:
- Answer ONLY from the provided context. Never hallucinate or invent facts.
- Always cite the source document name when referencing specific information.
- Use graph relationships to explain HOW concepts connect, not just WHAT they are.
- If the context does not contain enough information to answer, say so clearly.
- Be concise, accurate, and structured in your response."""


def build_user_prompt(query: str, context: str) -> str:
    return f"""CONTEXT:
{context}

QUESTION: {query}

Answer the question using only the context above. Cite sources where relevant."""


def _call_gemini(prompt: str) -> str:
    """Primary: Gemini 1.5 Flash — higher free tier limits than 2.5."""
    model = genai.GenerativeModel(
        "gemini-2.0-flash",           # Updated model name
        system_instruction=SYSTEM_PROMPT,
        generation_config={"temperature": 0.3, "max_output_tokens": 1024}
    )
    response = model.generate_content(prompt)
    return response.text


def _call_groq(prompt: str) -> str:
    """Fallback: Groq (llama-3.3-70b) — free, fast, no quota issues."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content


def generate_answer(query: str, context: str, citations: list) -> dict:
    """
    Try Gemini first. On any quota/rate error (429), fall back to Groq.
    Logs which model was actually used.
    """
    prompt = build_user_prompt(query, context)
    model_used = "gemini-2.0-flash"

    try:
        answer = _call_gemini(prompt)
        print(f"[LLM] Used: Gemini 2.0 Flash")

    except Exception as e:
        error_str = str(e)
        # catch quota exceeded, rate limit, or any Gemini API failure
        if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
            print(f"[LLM] Gemini quota hit — falling back to Groq")
            try:
                answer = _call_groq(prompt)
                model_used = "groq/llama-3.3-70b"
                print(f"[LLM] Used: Groq llama-3.3-70b (fallback)")
            except Exception as groq_err:
                print(f"[LLM] Groq also failed: {groq_err}")
                answer = "Both Gemini and Groq are currently unavailable. Please try again in a moment."
                model_used = "none"
        else:
            # non-quota Gemini error — still try Groq
            print(f"[LLM] Gemini error ({error_str[:80]}) — falling back to Groq")
            try:
                answer = _call_groq(prompt)
                model_used = "groq/llama-3.3-70b"
            except Exception as groq_err:
                answer = f"Service temporarily unavailable: {str(groq_err)[:100]}"
                model_used = "none"

    return {
        "answer":       answer,
        "citations":    citations,
        "context_used": bool(context),
        "model_used":   model_used       # useful for debugging + eval dashboard
    }