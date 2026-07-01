import google.generativeai as genai
import json, re, os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

EXTRACTION_PROMPT = """
Extract entities and relationships from this text.
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
{text}
"""

def extract_entities(chunk_text: str) -> dict:
    try:
        prompt = EXTRACTION_PROMPT.format(text=chunk_text[:2000])
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Entity extraction failed: {e}")
        return {"entities": [], "relationships": []}