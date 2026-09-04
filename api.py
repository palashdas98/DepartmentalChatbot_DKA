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
    version="4.1"
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
        "version": "4.1"
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

        is_fuel_economy_query = (
            "fuel economy" in lower_question
            or "mileage" in lower_question
            or "fe" in lower_question
        )

        if is_fuel_economy_query:

            search_query += """
            fuel economy
            mileage
            kmpl
            km/l
            FE
            fuel consumption
            laden
            unladen
            overall
            40 kmph
            55 kmph
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

        # Fuel-economy answers need more source rows (Unladen/
        # Laden/Overall x 40/55 kmph) than a typical question, and
        # those rows are often split across more chunks — widen
        # both the candidate pool and the reranked top_k for
        # those questions specifically.
        search_k = 30 if is_fuel_economy_query else 20

        results = vectorstore.similarity_search_with_score(
            search_query,
            k=search_k
        )

        retrieved_docs = [
            doc
            for doc, score in results
        ]

        # =============================================
        # BM25 RE-RANKING
        # =============================================

        rerank_top_k = 8 if is_fuel_economy_query else 5

        docs = bm25_rerank(
            search_query,
            retrieved_docs,
            top_k=rerank_top_k
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

{chunk[:2200]}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        # Protect against Groq token limits
        # (raised to fit the extra chunks pulled in for
        # fuel-economy questions; lower this back down if you hit
        # token-limit errors)

        context = context[:14000]

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
   - Report EVERY (condition, speed) data point that actually
     appears in the context — e.g. Unladen/Laden/Overall at
     40 kmph and/or 55 kmph — as its OWN table row. Do NOT force
     the data into a fixed grid of all conditions x all speeds.
   - If a particular condition/speed combination is not present
     in the context, simply omit that row. NEVER invent a row,
     and NEVER fill a missing value with "N/A" or leave a cell
     blank — a row only exists if the value exists.
   - Order the rows: Unladen first, then Laden, then Overall;
     within each condition, 40 kmph before 55 kmph.
   - Copy every numeric value EXACTLY as given in the context
     (same digits, same decimal places, same unit). Do not round,
     truncate, or shorten a value.
   - Only if L/Ton/100km (load-corrected fuel consumption)
     figures are present in the context, include them as their
     own short table or bullet list below the main table. If
     that data is not in the context, leave it out entirely —
     do not mention it or say it's unavailable.
   - Always write speed as "kmph" (e.g. "40 kmph", "55 kmph").
     NEVER use "km/h", "km h⁻¹", "kmh⁻¹", or any superscript/
     exponent notation for speed units.

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

7. FORMATTING (very important):
   - Whenever the answer contains several data points that vary
     by condition, speed, gear, or test run, present them as a
     "tidy" Markdown table: one row per data point, with a
     column for each distinguishing factor plus a value column.
   - Only include a row when every value in it is present in the
     context. Never pad a table with empty or "N/A" cells.
   - Always put a blank line before and after every table.
   - Use bullet points (not tables) for single, non-comparative
     values.

8. Never show:
   - PDF names
   - file names
   - page numbers
   - source names
   - citations

9. Give concise engineering answers, but do not omit a data
   point just to keep the answer short — completeness matters
   more than brevity when reporting comparative data.

10. Only reply:

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

        # Normalize every stray speed-unit variant (including
        # superscript/exponent forms like "km h⁻¹" or "km h-1")
        # down to a single plain "kmph".
        answer = re.sub(
            r"km\s*/?\s*h(?:\s*(?:⁻\s*1|-\s*1|\^-1))?",
            "kmph",
            answer,
            flags=re.IGNORECASE
        )

        # Only collapse repeated SPACES/TABS, never newlines, so
        # multi-line Markdown tables don't get squashed together.
        answer = re.sub(
            r"[ \t]{2,}",
            " ",
            answer
        )

        # Collapse 3+ blank lines down to a single blank line,
        # but keep the blank lines tables need to render.
        answer = re.sub(
            r"\n{3,}",
            "\n\n",
            answer
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