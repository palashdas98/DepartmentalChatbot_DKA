from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from llm import get_llm

# -------------------------------------------------
# Create FastAPI Application
# -------------------------------------------------

app = FastAPI(
    title="Departmental RAG Chatbot",
    description="RAG API using FAISS + Groq",
    version="1.0"
)

# -------------------------------------------------
# Load Vector Database Once at Startup
# -------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

# -------------------------------------------------
# Input Schema
# -------------------------------------------------

class QuestionRequest(BaseModel):
    question: str

# -------------------------------------------------
# Health Check
# -------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Departmental Chatbot API Running"
    }

# -------------------------------------------------
# Chat Endpoint
# -------------------------------------------------

@app.post("/chat")
def chat(request: QuestionRequest):

    user_question = request.question

    docs = vectorstore.similarity_search(
        user_question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {user_question}
    """

    llm = get_llm()

    response = llm.invoke(prompt)

    return {
        "question": user_question,
        "answer": response
    }