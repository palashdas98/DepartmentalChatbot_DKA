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
    version="4.2"
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
# MODEL-CODE EXACT MATCH
# =====================================================
# Pure embedding similarity can silently fail to shortlist the
# right chunk at all (a question phrased slightly differently can
# miss data that a near-identical phrasing finds) because the two
# phrasings embed differently. Since Tata Motors vehicle model
# codes (e.g. "3522.T", "3023.T", "4221.T") are exact, unambiguous
# strings, we scan the WHOLE vectorstore for a literal match
# whenever the question names one — independent of embedding
# similarity. This guarantees every chunk about that vehicle is
# found, however many chunks its data got split across.

def extract_model_codes(text):
    pattern = r'\b(\d{3,4})\s*\.?\s*T\b'
    codes = set()
    for num in re.findall(pattern, text, flags=re.IGNORECASE):
        codes.add(f"{num}.T")
    return list(codes)


def get_all_vectorstore_docs(vs):
    # Best-effort access to every chunk stored in the FAISS index
    # via langchain's in-memory docstore. If this internal
    # structure ever changes in a future langchain version, we
    # fall back to an empty list rather than erroring out, and the
    # API just behaves as it did before this fix.
    try:
        return list(vs.docstore._dict.values())
    except AttributeError:
        return []


def find_docs_by_model_code(vs, model_codes):
    if not model_codes:
        return []
    all_docs = get_all_vectorstore_docs(vs)
    matched = []
    for doc in all_docs:
        content_compact = re.sub(
            r"\s+", "", doc.page_content
        ).upper()
        for code in model_codes:
            code_compact = code.replace(" ", "").upper()
            if code_compact in content_compact:
                matched.append(doc)
                break
    return matched


# =====================================================
# ROOT
# =====================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "application": "Department Knowledge Assistant API",
        "version": "4.2"
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

        # "fe" and "pd" are short enough that a plain substring
        # check misfires on ordinary words ("fe" inside
        # "reference" or "perfect"; "pd" inside "updated") — use
        # word boundaries so only the standalone abbreviation
        # triggers the branch.
        is_fuel_economy_query = (
            "fuel economy" in lower_question
            or "fluid economy" in lower_question
            or "mileage" in lower_question
            or re.search(r'\bfe\b', lower_question) is not None
        )

        # Performance / Acceleration / PD (Performance
        # Drivability) all share one standardized answer format
        # (see prompt rule 5 below), so they're detected and
        # retrieved identically.
        is_pd_query = (
            "performance" in lower_question
            or "acceleration" in lower_question
            or "drivability" in lower_question
            or "pd summary" in lower_question
            or re.search(r'\bpd\b', lower_question) is not None
        )

        if is_fuel_economy_query:

            search_query += """
            fuel economy
            fluid economy
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
            test vehicle
            """

        elif is_pd_query:

            search_query += """
            performance
            acceleration
            0-60 kmph
            0-1000 m
            WOT
            driveability
            panel chart
            bath-tub drivability
            bathtub drivability
            gear
            speed band
            test vehicle
            sustain gradeability
            restart gradeability
            """

        model_codes = extract_model_codes(question)

        # =============================================
        # RETRIEVAL
        # =============================================

        # Exact, embedding-independent match on any model code
        # mentioned in the question.
        exact_docs = find_docs_by_model_code(
            vectorstore,
            model_codes
        )

        search_k = (
            30
            if (is_fuel_economy_query or is_pd_query)
            else 20
        )

        results = vectorstore.similarity_search_with_score(
            search_query,
            k=search_k
        )

        semantic_docs = [
            doc
            for doc, score in results
        ]

        # When the question names a model code and we found exact
        # matches for it, use ONLY those — mixing in semantic
        # candidates risks pulling in a different vehicle's report
        # (e.g. a similarly-worded doc for a different model
        # number) that eats into the context budget and can
        # distract the LLM into citing the wrong vehicle's
        # figures. Only fall back to the semantic candidates when
        # no model code was found (or it found nothing), so
        # general questions still work as before.
        source_docs = exact_docs if exact_docs else semantic_docs

        seen_keys = set()
        deduped_docs = []
        for doc in source_docs:
            key = (
                doc.metadata.get("source", "Unknown"),
                doc.metadata.get("page", "N/A"),
                doc.page_content[:80]
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_docs.append(doc)

        # =============================================
        # BM25 RE-RANKING
        # =============================================

        # When we found exact model-code matches, trust them
        # fully: BM25-rerank only decides their ORDER, it never
        # drops one — each already names the vehicle asked about.
        # Only fall back to a smaller top_k when there's no model
        # code to anchor on.
        if exact_docs:
            rerank_top_k = min(len(deduped_docs), 14)
        elif is_fuel_economy_query or is_pd_query:
            rerank_top_k = 8
        else:
            rerank_top_k = 5

        docs = bm25_rerank(
            search_query,
            deduped_docs,
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

        # Stop adding chunks once the budget is reached rather
        # than slicing the final joined string — slicing mid
        # string could cut a chunk (and a table row) in half;
        # stopping at a whole chunk boundary never does, and keeps
        # the highest-priority chunks fully intact.
        #
        # Groq's free/on_demand tier for openai/gpt-oss-20b caps
        # total tokens per request (prompt + completion) at 8000
        # TPM. 9000 characters of context keeps the full request
        # (rules text + system prompt + context + completion)
        # safely under that cap — see llm.py's max_tokens comment
        # for the full budget breakdown. Raising this risks the
        # "Request too large" 413 error unless max_tokens there is
        # lowered to compensate.

        MAX_CONTEXT_CHARS = 9000

        context_parts = []
        seen_chunks = set()
        total_len = 0

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

            piece = f"""
DOCUMENT: {source}

PAGE: {page}

{chunk[:2200]}
"""

            if total_len + len(piece) > MAX_CONTEXT_CHARS:
                break

            context_parts.append(piece)
            total_len += len(piece)

        context = "\n\n".join(
            context_parts
        )

        # =============================================
        # PROMPT
        # =============================================

        prompt = f"""
You are Tata Motors Department Knowledge Assistant.

Rules:

1. Use ONLY the supplied context.

2. Extract ALL available numerical values.

3. Never ignore values present in tables.

4. For Fuel/Fluid Economy questions:
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
   - If the source table ALSO shows other columns such as
     Target, PALS, Benchmark, Base Vehicle, or Competitor, apply
     the same column rule as Performance/PD below: use ONLY the
     "Test Vehicle" column's values.

5. For Performance / Acceleration / PD (Performance Drivability)
   questions, standardize the answer exactly as follows:

   a) SINGLE-SOURCE RULE (most important - prevents mixing data
      from different tables):
      - If more than one performance table for this vehicle
        appears in the context (e.g. a Panel Chart AND a separate
        "Performance Data Summary" / Annexure table), pick ONE
        table - the one with an explicit "Test Vehicle" (or
        "Current/Gateway" / "Status") column - and take EVERY
        value in this answer from that SAME table only.
      - NEVER combine a value from one table/page with a value
        from a different table/page for the same answer, even if
        both mention the same vehicle. If uncertain which table
        to use, prefer the one with the most complete data for
        the fields below.

   b) COLUMN RULE (which column to read, within that one table):
      - ALWAYS read values from the vehicle's own "Test Vehicle"
        (or "Current/Gateway" / "Status") column.
      - NEVER use "Target", "PALS", "Benchmark", "Base Vehicle",
        or "Competitor" column values in the answer - those
        describe a threshold or a different vehicle, not the
        vehicle being asked about.
      - If the Test Vehicle column has more than one labelled
        sub-condition (e.g. "RDE HM", "RDE BM", "RDE LM", or
        similar mode suffixes), report the FIRST (left-most) one
        by default. Only report a different one if the question
        explicitly names that mode (e.g. "BM", "Base Mode").
      - A relative statement ("X% inferior/superior to base
        vehicle") may be added as EXTRA context after an absolute
        value, but must NEVER replace the absolute value.

   c) COLUMN-SWAP SAFEGUARD (apply this check to every row before
      writing it - this is how to tell the columns apart even if
      headers are unclear or jumbled in the extracted text):
      - The Target / PALS value is a pass/fail THRESHOLD and is
        ALWAYS written with a "≤" or "≥" symbol (e.g. "≤66.30",
        "≥22.04") - never a bare number.
      - The actual measured Test Vehicle / Current-Gateway /
        Status result is ALWAYS a plain number with NO "≤" or "≥"
        symbol (e.g. "65.49", "22.56").
      - If you are about to put a value WITH a "≤"/"≥" symbol into
        a "Result" / "Test Vehicle" / "Current-Gateway" column or
        field, STOP - that value belongs in the Target column
        instead, and you have the two swapped. Re-check and
        correct before writing the final answer.
      - When presenting a Result-vs-Target comparison table, the
        "Result" column must contain ONLY plain numbers (no ≤/≥),
        and the "Target"/"Benchmark" column is the only one that
        may contain a ≤/≥ symbol.

   d) WHAT TO INCLUDE, when present in the chosen table's Test
      Vehicle column:
      - 0-60 kmph and 0-1000 m acceleration times, in seconds.
      - Gear-wise times from the "Bath-Tub Drivability" (or
        "Bathtub Driveability") table - this breaks each gear
        into its own speed bands (e.g. "3rd Gear (10-15 kmph)",
        "3rd Gear (15-20 kmph)", "4th Gear (20-25 kmph)", and so
        on). Report EVERY gear/speed-band row that exists in the
        context - this may run from 3rd Gear through 6th, 7th, or
        8th Gear depending on the document; never assume it stops
        at 6th, and never omit a row that is present.
      - Do NOT report the simple "Rate of Acceleration" table
        (one plain value per gear, e.g. "1st Gear: 2.26 kmph/s",
        no speed bands) UNLESS the Bath-Tub Drivability table is
        not present anywhere in the context for this vehicle -
        only then use Rate of Acceleration so gear data isn't
        omitted entirely from the answer.
      - Sustain Gradeability (%) and Restart Gradeability (%).

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
   more than brevity when reporting comparative or panel-chart
   data.

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

        answer = re.sub(
            r"km\s*/?\s*h(?:\s*(?:⁻\s*1|-\s*1|\^-1))?",
            "kmph",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"[ \t]{2,}",
            " ",
            answer
        )

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

            "model_codes_detected": model_codes,

            "retrieved_chunks": len(docs),

            "sources": retrieved_sources

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
