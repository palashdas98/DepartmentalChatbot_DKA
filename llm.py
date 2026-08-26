import os

from groq import Groq
from dotenv import load_dotenv

# ======================================================
# LOAD ENV
# ======================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in Environment Variables"
    )

# ======================================================
# CREATE CLIENT ONCE
# ======================================================

client = Groq(
    api_key=GROQ_API_KEY
)

# ======================================================
# LLM WRAPPER
# ======================================================

class LLMWrapper:

    def invoke(self, prompt):

        try:

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """
You are a Tata Motors Department Knowledge Assistant.

Rules:
1. Use only provided context.
2. Never hallucinate.
3. Mention values exactly.
4. Mention source document if available.
5. If answer not found say:
Information not found in documents.
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:

            print("GROQ ERROR:")
            print(str(e))

            raise Exception(
                f"Groq API Error: {str(e)}"
            )

# ======================================================
# GET MODEL
# ======================================================

def get_llm():
    return LLMWrapper()