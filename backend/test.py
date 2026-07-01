import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

print("API KEY:", os.getenv("GEMINI_API_KEY")[:10] + "...")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

try:
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(
        "Say Hello"
    )

    print(response.text)

except Exception as e:
    print(type(e))
    print(e)