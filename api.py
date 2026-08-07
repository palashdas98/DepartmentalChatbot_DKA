from collections import defaultdict
import traceback

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
# LOAD VECTOR DATABASE
# =====================================================

print("Loading Vector Database...")

vectorstore = None
def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.load_local(
            "vectorstore",
            embeddings,
            allow_dangerous_deserialization=True
        )
    return vectorstore

print("Vector Database Loaded Successfully")

# =====================================================
# ROOT ENDPOINT
# =====================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Department Knowledge Assistant API"
    }

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# =====================================================
# CHAT ENDPOINT
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

        # =============================================
        # QUERY EXPANSION
        # =============================================

        expanded_query = f"""
Question: {question}

Synonyms:
Fuel Economy
FLE
FE
Mileage
Fuel Consumption
Road Speed
Road Speed Governor
Vehicle Speed
Maximum Speed
BSFC
Performance
"""

        # =============================================
        # RETRIEVAL
        # =============================================

        db = get_vectorstore()

        docs = db.max_marginal_relevance_search(
            expanded_query,
            k=5,
            fetch_k=15
        )

        if len(docs) == 0:

            return {
                "question": question,
                "answer": "Information not found in documents.",
                "sources": []
            }

        # =============================================
        # BUILD CONTEXT
        # =============================================

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

===================================================
"""

        # =============================================
        # PRIMARY SOURCE
        # =============================================

        top_doc = docs[0]

        primary_source = top_doc.metadata.get(
            "source",
            "Unknown"
        )

        primary_page = top_doc.metadata.get(
            "page",
            "N/A"
        )

        # =============================================
        # PROMPT
        # =============================================

        prompt = f"""
You are a Tata Motors departmental knowledge assistant.

Instructions:

1. Use ONLY the supplied context.
2. Never invent information.
3. Combine information from multiple chunks when required.
4. Mention exact values and units.
5. Mention exact limits and specifications.
6. If answer exists partially, provide available information.
7. Use bullet points whenever possible.
8. Mention source document and page number.
9. If answer is not available, say:
   'Information not found in documents.'

CONTEXT:

{context}

QUESTION:

{question}

ANSWER FORMAT:

Answer:
<answer>

Source:
<source>

Page:
<page>
"""

        # =============================================
        # CALL LLM
        # =============================================

        llm = get_llm()

        answer = llm.invoke(prompt)

        # =============================================
        # COLLECT SOURCES
        # =============================================

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
                    "pages": sorted(
                        list(pages)
                    )
                }
            )

        # =============================================
        # RESPONSE
        # =============================================

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
        
        print("ERROR:")

        print(traceback.format_exc())
    
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )