import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "openai/gpt-oss-120b"


class LLMWrapper:

    def invoke(self, context, question):

        prompt = f"""
You are Tata Motors Department Knowledge Assistant.

Instructions:
1. Use only the provided context.
2. Never create information.
3. If answer exists, provide exact values.
4. Mention source document.
5. If information is absent, say:
Information not found in documents.

CONTEXT:
{context}

QUESTION:
{question}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content


def get_llm():
    return LLMWrapper()