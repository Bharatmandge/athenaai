from groq import Groq
import os, json, re
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACTION_PROMPT = """Extract entities and relationships from this text.
Return ONLY valid JSON, no explanation, no markdown fences.

Format:
{{
  "entities": [
    {{"name": "entity name", "type": "PERSON|ORG|TECH|PLACE|OTHER"}}
  ],
  "relationships": [
    {{"source": "entity A", "relation": "RELATED_TO", "target": "entity B"}}
  ]
}}

Text:
{text}"""


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


def _extract_with_gemini(chunk_text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(text=chunk_text[:2000])
    response = client_gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return _parse_json(response.text)


def _extract_with_groq(chunk_text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(text=chunk_text[:2000])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,       # very low — we want structured JSON, not creativity
        max_tokens=1024
    )
    return _parse_json(response.choices[0].message.content)


def extract_entities(chunk_text: str) -> dict:
    """Gemini primary → Groq fallback for entity extraction."""
    # try Gemini first
    try:
        result = _extract_with_gemini(chunk_text)
        print("[Extractor] Used: Gemini")
        return result
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            print("[Extractor] Gemini quota hit — switching to Groq")
        else:
            print(f"[Extractor] Gemini error — switching to Groq: {error_str[:60]}")

    # fallback to Groq
    try:
        result = _extract_with_groq(chunk_text)
        print("[Extractor] Used: Groq (fallback)")
        return result
    except Exception as groq_err:
        print(f"[Extractor] Groq also failed: {groq_err}")
        return {"entities": [], "relationships": []}   # never crash the pipeline