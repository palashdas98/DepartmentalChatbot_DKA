import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in Environment Variables"
    )

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "openai/gpt-oss-20b"


class LLMWrapper:

    def invoke(self, prompt):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": """
You are Tata Motors Department Knowledge Assistant.

Rules:

1. Use ONLY the supplied context.
2. Never hallucinate.
3. Give exact values from documents, copied exactly
   (same digits, same units) — never round or shorten a value.
4. Extract data correctly from tables.
5. When the context contains multiple related data points
   (e.g. several conditions, speeds, or gears), report ALL of
   them — completeness matters more than brevity here. Keep the
   answer direct and free of unnecessary commentary, but do not
   omit a data point just to keep the answer short.
6. If information is not available, reply exactly:

Information not found in documents.
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=2000
            )

            return (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

        except Exception as e:

            print("\n========== GROQ ERROR ==========")
            print(str(e))
            print("================================\n")

            raise Exception(
                f"Groq API Error: {str(e)}"
            )


def get_llm():
    return LLMWrapper()