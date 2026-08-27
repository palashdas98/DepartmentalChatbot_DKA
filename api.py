from collections import defaultdict
import re

from fastapi import FastAPI
from fastapi import HTTPException

from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm import get_llm
from hybrid_retriever import bm25_rerank


# =====================================================
# FASTAPI CONFIG
# =====================================================

app = FastAPI(
    title="Department Knowledge Assistant API",
    description="RAG-based PDF Question Answering System",
    version="4.0"
)


# =====================================================
# REQUEST MODEL
# =====================================================

class QuestionRequest(BaseModel):
    question: str


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

print("Loading Embedding Model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

print("Embedding Model Loaded")


# =====================================================
# LOAD VECTORSTORE
# =====================================================

print("Loading Vector Store...")

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Vector Store Loaded Successfully")


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "application": "Department Knowledge Assistant API",
        "version": "4.0"
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

        # =============================================
        # QUERY EXPANSION
        # =============================================

        search_query = question

        lower_question = question.lower()

        if (
            "fuel economy" in lower_question
            or "mileage" in lower_question
            or "fe" in lower_question
        ):

            search_query += """
            fuel economy
            mileage
            kmpl
            km/l
            FE
            fuel consumption
            """

        elif "acceleration" in lower_question:

            search_query += """
            acceleration
            0-60 kmph
            0-1000 m
            WOT
            driveability
            """

        elif "performance" in lower_question:

            search_query += """
            acceleration
            fuel economy
            driveability
            WOT
            restart gradeability
            sustainability
            gradeability
            """

        # =============================================
        # VECTOR SEARCH
        # =============================================

        results = vectorstore.similarity_search_with_score(
            search_query,
            k=20
        )

        retrieved_docs = [
            doc
            for doc, score in results
        ]

        # =============================================
        # BM25 RE-RANKING
        # =============================================

        docs = bm25_rerank(
            search_query,
            retrieved_docs,
            top_k=5
        )

        if not docs:

            return {
                "question": question,
                "answer": "Information not found in documents.",
                "sources": []
            }

        # =============================================
        # BUILD CONTEXT
        # =============================================

        context_parts = []
        seen_chunks = set()

        for doc in docs:

            chunk = doc.page_content.strip()

            if chunk in seen_chunks:
                continue

            seen_chunks.add(chunk)

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            context_parts.append(
                f"""
DOCUMENT: {source}

PAGE: {page}

{chunk[:1200]}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # Protect against Groq 413 errors

        context = context[:6000]

        # =============================================
        # PROMPT
        # =============================================

        prompt = f"""
You are Tata Motors Department Knowledge Assistant.

Rules:

1. Use ONLY the supplied context.

2. Extract ALL available numerical values.

3. Never ignore values present in tables.

4. For Fuel Economy questions:
   - Provide exact FE values.
   - Provide mileage values.
   - Mention test conditions if available.

5. For Performance questions:
   Include:
   - 0-60 kmph
   - 0-1000 m
   - Acceleration
   - WOT Driveability
   - Restart Gradeability
   - Sustainability

6. For gear-wise values use:

• 1st Gear : value
• 2nd Gear : value
• 3rd Gear : value
• 4th Gear : value
• 5th Gear : value
• 6th Gear : value

7. Never show:
   - PDF names
   - file names
   - page numbers
   - source names
   - citations

8. Give concise engineering answers.

9. Only reply:

Information not found in documents.

if absolutely no relevant information exists.

CONTEXT:

{context}

QUESTION:

{question}
"""

        # =============================================
        # LLM CALL
        # =============================================

        llm = get_llm()

        answer = llm.invoke(
            prompt
        )

        if (
            not answer
            or not answer.strip()
        ):

            answer = (
                "Relevant documents were retrieved "
                "but no answer could be generated."
            )

        # =============================================
        # ANSWER CLEANUP
        # =============================================

        answer = re.sub(
            r"\[.*?\]",
            "",
            answer
        )

        answer = re.sub(
            r"source\s*page\s*\d+",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"<br\s*/?>",
            "\n",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"<[^>]+>",
            "",
            answer
        )

        answer = answer.replace(
            "1st gear:",
            "\n• 1st Gear : "
        )

        answer = answer.replace(
            "2nd gear:",
            "\n• 2nd Gear : "
        )

        answer = answer.replace(
            "3rd gear:",
            "\n• 3rd Gear : "
        )

        answer = answer.replace(
            "4th gear:",
            "\n• 4th Gear : "
        )

        answer = answer.replace(
            "5th gear:",
            "\n• 5th Gear : "
        )

        answer = answer.replace(
            "6th gear:",
            "\n• 6th Gear : "
        )

        answer = answer.replace(
            "km/h/s",
            "Kmph/Sec"
        )

        answer = answer.replace(
            "km/h·s⁻¹",
            "Kmph/Sec"
        )

        answer = answer.strip()

        # =============================================
        # SOURCES
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

            "retrieved_chunks": len(docs),

            "sources": retrieved_sources

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )