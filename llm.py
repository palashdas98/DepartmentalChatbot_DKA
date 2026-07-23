from groq import Groq
from dotenv import load_dotenv
import os

# ===========================================
# LOAD ENVIRONMENT VARIABLES
# ===========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )

# ===========================================
# GROQ CLIENT
# ===========================================

client = Groq(
    api_key=api_key
)

# ===========================================
# LLM WRAPPER
# ===========================================

class LLMWrapper:

    def invoke(self, prompt):

        try:

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Tata Motors departmental assistant. "
                            "Answer only from the supplied context. "
                            "If the answer exists in the context, summarize it clearly. "
                            "Do not invent information."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:

            return f"Groq Error: {str(e)}"

# ===========================================
# GET LLM
# ===========================================

def get_llm():
    return LLMWrapper()