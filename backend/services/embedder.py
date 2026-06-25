from dotenv import load_dotenv
import google.generativeai as genai
import os
import time

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def embed_text(
    text: str,
    task_type="retrieval_document"
):
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task_type,
        output_dimensionality=768
    )

    return response["embedding"]


def embed_batch(texts):
    embeddings = []

    for i, text in enumerate(texts):
        embeddings.append(embed_text(text))

        if i % 10 == 9:
            time.sleep(1)

    return embeddings