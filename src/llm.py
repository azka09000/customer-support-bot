import os
from dotenv import load_dotenv
from pathlib import Path
from google import genai

# ---------------------------
# Load .env safely
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


class GeminiLLM:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def generate_answer(self, question, context_chunks, history=None):
        """
        Strict RAG-based answer generation
        """
        history = history or ""

        # Format retrieved context
        formatted_context = "\n\n".join(
            f"[Chunk {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        # ---------------- STRICT PROMPT ----------------
        prompt = f"""
You are a strict document-based AI customer support assistant.

RULES:
- Use ONLY the provided context
- Do NOT use external knowledge
- Do NOT guess or assume missing information
- If answer is not in context, say: "Not found in documents"
- Keep answers short, clear, and structured
- Prefer bullet points (max 5-7 points)
- Avoid repetition

OUTPUT FORMAT:
- Bullet points only
- No headings
- No extra explanation

CONTEXT:
{formatted_context}

CHAT HISTORY:
{history if history.strip() else "None"}

QUESTION:
{question}

FINAL ANSWER:
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text