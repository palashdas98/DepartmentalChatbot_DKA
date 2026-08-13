from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm import get_llm

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Department Knowledge Assistant",
    description="RAG Chatbot API using FAISS + Groq",
    version="1.0"
)

# =====================================================
# REQUEST MODEL
# =====================================================

class QuestionRequest(BaseModel):
    question: str

# =====================================================
# LOAD EMBEDDINGS ONCE
# =====================================================

print("Loading Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================================
# LOAD FAISS VECTORSTORE ONCE
# =====================================================

print("Loading Vector Database...")

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Vector Database Loaded Successfully")

# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Department Knowledge Assistant API"
    }

# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# =====================================================
# CHAT
# =====================================================

@app.post("/chat")
def chat(request: QuestionRequest):

    try:

        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        print(f"Question Received: {question}")

        # ===========================================
        # RETRIEVE RELEVANT DOCUMENTS
        # ===========================================

        results = vectorstore.similarity_search_with_score(
            question,
            k=4
        )

        if not results:
            return {
                "question": question,
                "answer": "Information not found in documents.",
                "sources": []
            }

        docs = [doc for doc, score in results]

        # ===========================================
        # BUILD CONTEXT
        # ===========================================

        context = ""

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            context += f"""
SOURCE DOCUMENT: {source}

PAGE NUMBER: {page}

{doc.page_content}

=================================================
"""

        # ===========================================
        # TOP DOCUMENT
        # ===========================================

        top_doc = docs[0]

        primary_source = top_doc.metadata.get(
            "source",
            "Unknown"
        )

        primary_page = top_doc.metadata.get(
            "page",
            "N/A"
        )

        # ===========================================
        # PROMPT
        # ===========================================

        prompt = f"""
You are a Tata Motors Department Knowledge Assistant.

Use ONLY the context below.

If answer is unavailable, say:
Information not found in documents.

CONTEXT:

{context}

QUESTION:

{question}
"""

        # ===========================================
        # LLM RESPONSE
        # ===========================================

        llm = get_llm()

        answer = llm.invoke(prompt)

        # ===========================================
        # SOURCES
        # ===========================================

        source_pages = defaultdict(set)

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            source_pages[source].add(page)

        retrieved_sources = []

        for source, pages in source_pages.items():

            retrieved_sources.append(
                {
                    "source": source,
                    "pages": sorted(list(pages))
                }
            )

        # ===========================================
        # RESPONSE
        # ===========================================

        return {
            "question": question,
            "answer": answer,
            "primary_source": primary_source,
            "primary_page": primary_page,
            "retrieved_chunks": len(docs),
            "retrieved_sources": retrieved_sources
        }

    except Exception as e:

        import traceback

        print("===================================")
        print("CHAT ERROR")
        print(traceback.format_exc())
        print("===================================")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )