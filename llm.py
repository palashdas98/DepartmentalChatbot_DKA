import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "openai/gpt-oss-120b"


class LLMWrapper:

    def invoke(self, context, question):

        prompt = f"""
You are Tata Motors Department Knowledge Assistant.

<<<<<<< HEAD
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
3. Give exact values from documents.
4. Extract data correctly from tables.
5. Give concise and direct answers.
6. If information is not available, reply exactly:

=======
Instructions:
1. Use only the provided context.
2. Never create information.
3. If answer exists, provide exact values.
4. Mention source document.
5. If information is absent, say:
>>>>>>> 5943ed513b498b440d438f7a2c7498d1333299cd
Information not found in documents.

CONTEXT:
{context}

QUESTION:
{question}
"""

<<<<<<< HEAD
            return (
                response
                .choices[0]
                .message
                .content
                .strip()
            )
=======
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
>>>>>>> 5943ed513b498b440d438f7a2c7498d1333299cd

        return response.choices[0].message.content

<<<<<<< HEAD
            print("\n========== GROQ ERROR ==========")
            print(str(e))
            print("================================\n")

            raise Exception(
                f"Groq API Error: {str(e)}"
            )

=======
>>>>>>> 5943ed513b498b440d438f7a2c7498d1333299cd

def get_llm():
    return LLMWrapper()