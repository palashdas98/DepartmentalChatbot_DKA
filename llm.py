from groq import Groq
from dotenv import load_dotenv
import os

# ===========================================
# LOAD ENVIRONMENT VARIABLES
# ===========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        pass
if not api_key:
    raise ValueError("GROQ_API_KEY not found")        

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
                        "content": """
You are a Tata Motors departmental knowledge assistant.

Rules:
1. Answer only using the supplied context.
2. Never hallucinate.
3. Mention exact values and limits.
4. Mention source document if available.
5. Combine information across chunks.
6. If answer is not found, clearly say:
   'Information not found in documents.'
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

            return response.choices[0].message.content

        except Exception as e:

            return f"Groq Error: {str(e)}"


# ===========================================
# RETURN MODEL
# ===========================================

def get_llm():
    return LLMWrapper()
