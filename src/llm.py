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
        formatted_context = "\n\n".join(
            f"[Chunk {i+1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )

        prompt = f"""You are an expert assistant answering questions about university regulations and academic policies.

INSTRUCTIONS:
1. Use ONLY the provided context to answer questions
2. If the answer is not available in the context, say: "I cannot find this information in the provided documents."
3. Structure your answer clearly in paragraphs
4. Cite the relevant chunk when making claims (e.g., "According to Chunk 1...")
5. Be concise but comprehensive
6. If the question is ambiguous, ask for clarification

CONTEXT DOCUMENTS:
{formatted_context}

PREVIOUS CONVERSATION:
{history if history else "None"}

QUESTION FROM USER:
{question}

ANSWER:"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text