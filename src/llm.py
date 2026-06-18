import os
import google.generativeai as genai


class GeminiLLM:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")

        genai.configure(api_key=api_key)

        # IMPORTANT: use a valid model from your list_models output
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_answer(self, question, context_chunks):
        context = "\n\n".join(context_chunks)

        prompt = f"""
You are a helpful and precise assistant.

RULES:
- Use ONLY the provided context.
- If the answer is not in the context, say "I cannot find this in the provided context."
- Keep the answer concise and factual.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"Error generating response: {str(e)}"