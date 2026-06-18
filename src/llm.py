import os
from google import genai


class GeminiLLM:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def generate_answer(self, question, context_chunks, history=""):
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are a helpful assistant.

Use ONLY the context to answer.

Previous conversation:
{history}

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text