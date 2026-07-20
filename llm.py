from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def ask_gpt(question, context):

    prompt = f"""
    Answer ONLY from the supplied context.

    Context:
    {context}

    Question:
    {question}

    If the answer is not found in the context,
    reply:

    Information not found in the documents.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content