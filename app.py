import re
import streamlit as st
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm import get_llm
from hybrid_retriever import bm25_rerank

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Department Knowledge Assistant",
    page_icon="🚛",
    layout="wide"
)
st.title("Department Knowledge Assistant")
st.caption(
    "AI-powered document retrieval and question answering system"
)

# =====================================================
# LOAD VECTORSTORE
# =====================================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

try:
    vectorstore = load_vectorstore()
    st.success(
        "✅ Vector Database Loaded Successfully"
    )
except Exception as e:
    st.error(
        f"Vector Database Load Error: {e}"
    )
    st.stop()


# =====================================================
# MODEL-CODE EXACT MATCH
# =====================================================
# Pure embedding similarity can silently fail to shortlist the
# right chunk at all (e.g. "Acceleration of signa 3522.T CNG"
# missed data that "Test vehicle Acceleration of Signa 3522.T CNG
# vehicle" found) because the two phrasings embed slightly
# differently. Since Tata Motors vehicle model codes (e.g.
# "3522.T", "3023.T", "4221.T") are exact, unambiguous strings,
# we scan the WHOLE vectorstore for a literal match whenever the
# question names one — independent of embedding similarity. This
# guarantees every chunk about that vehicle is found, however many
# chunks its data got split across.

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
    # app just behaves as it did before this fix.
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
# DOCUMENT-TYPE FILTER
# =====================================================
# A vehicle model code (e.g. "3525.T") can appear in MULTIPLE
# different reports for the same vehicle — a Performance
# evaluation PDF AND a separate Fuel/Fluid Economy evaluation PDF.
# Exact model-code matching alone can't tell these apart, so a
# pure Performance question was pulling in Fluid Economy data (and
# vice versa) just because both documents mention the same model
# code. These report types name themselves in their own filenames
# (e.g. "...Performance evaluation on Signa 3525.T...pdf" vs
# "...Fluid Economy evaluation on Signa 3525.T...pdf"), so filter
# by filename once the question's intent (FE vs PD) is known.
FE_FILENAME_KEYWORDS = ["fuel economy", "fluid economy"]
PD_FILENAME_KEYWORDS = ["performance"]


def filter_docs_by_report_type(docs, is_fuel_economy_query, is_pd_query):
    # Only filter when the question is clearly about ONE report
    # type and not the other — an ambiguous/general question is
    # left unfiltered.
    if is_fuel_economy_query and not is_pd_query:
        keywords = FE_FILENAME_KEYWORDS
    elif is_pd_query and not is_fuel_economy_query:
        keywords = PD_FILENAME_KEYWORDS
    else:
        return docs

    filtered = [
        doc
        for doc in docs
        if any(
            kw in doc.metadata.get("source", "").lower()
            for kw in keywords
        )
    ]
    # If filename-based filtering happens to eliminate everything
    # (e.g. a PDF that doesn't follow this naming convention),
    # fall back to the unfiltered list rather than returning
    # nothing for a vehicle that genuinely has data.
    return filtered if filtered else docs


# =====================================================
# STRUCTURED-TABLE PRIORITIZATION
# =====================================================
# Within a Performance report, the authoritative numbers live in
# the Panel Chart / "Performance Data Summary" attribute table —
# not in narrative prose elsewhere in the same PDF (which can
# contain only partial or relative figures). Split retrieved
# chunks into "structured" (contains a recognizable attribute-table
# marker) and "other", then rerank and fill the chunk budget from
# the structured group FIRST, only falling back to "other" chunks
# if there aren't enough structured ones to fill it.
STRUCTURED_TABLE_MARKERS = [
    "panel chart",
    "current/gateway",
    "current / gateway",
    "performance data summary",
    "bath-tub",
    "bathtub",
]


def prioritize_structured_chunks(search_query, docs, top_k):
    structured = [
        doc
        for doc in docs
        if any(
            marker in doc.page_content.lower()
            for marker in STRUCTURED_TABLE_MARKERS
        )
    ]
    other = [
        doc
        for doc in docs
        if doc not in structured
    ]

    structured_ranked = bm25_rerank(
        search_query,
        structured,
        top_k=top_k
    )
    remaining = top_k - len(structured_ranked)
    other_ranked = (
        bm25_rerank(search_query, other, top_k=remaining)
        if remaining > 0
        else []
    )
    return structured_ranked + other_ranked


# =====================================================
# USER INPUT
# =====================================================
question = st.text_input(
    "Ask a question from your PDFs"
)
if question:
    # =================================================
    # QUERY EXPANSION
    # =================================================
    search_query = question
    lower_question = question.lower()
    # "fe" and "pd" are short enough that a plain substring check
    # misfires on ordinary words ("fe" inside "reference" or
    # "perfect"; "pd" inside "updated") — use word boundaries so
    # only the standalone abbreviation triggers the branch.
    is_fuel_economy_query = (
        "fuel economy" in lower_question
        or "fluid economy" in lower_question
        or "mileage" in lower_question
        or re.search(r'\bfe\b', lower_question) is not None
    )
    # Performance / Acceleration / PD (Performance Drivability) all
    # share one standardized answer format (see prompt rule 5
    # below), so they're detected and retrieved identically.
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
        FE
        kmpl
        km/l
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

    # =================================================
    # RETRIEVAL
    # =================================================
    with st.spinner(
        "Searching Knowledge Base..."
    ):
        # Exact, embedding-independent match on any model code
        # mentioned in the question.
        exact_docs = find_docs_by_model_code(
            vectorstore,
            model_codes
        )

        # Fuel-economy / PD (performance/acceleration) tables often
        # get split across more chunks than other topics, so
        # widen the semantic candidate pool for those questions.
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
    # (e.g. a similarly-worded doc for a different model number)
    # that eats into the context budget and can distract the LLM
    # into citing the wrong vehicle's figures. Only fall back to
    # the semantic candidates when no model code was found (or it
    # found nothing), so general questions still work as before.
    source_docs = exact_docs if exact_docs else semantic_docs

    # Same model code can appear in both a Performance report and
    # a separate Fuel/Fluid Economy report for that vehicle — keep
    # only the report type the question is actually asking about.
    source_docs = filter_docs_by_report_type(
        source_docs,
        is_fuel_economy_query,
        is_pd_query
    )

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

    # When we found exact model-code matches, trust them fully:
    # BM25-rerank only decides their ORDER (most relevant chunk
    # first), it never drops one — each already names the vehicle
    # asked about and may hold the only copy of a particular row
    # (a fuel-economy condition, an acceleration figure, a panel
    # chart). Only fall back to a smaller top_k when there's no
    # model code to anchor on.
    if exact_docs:
        rerank_top_k = min(len(deduped_docs), 14)
    elif is_fuel_economy_query or is_pd_query:
        rerank_top_k = 8
    else:
        rerank_top_k = 5

    if is_pd_query:
        # For performance questions, fill the chunk budget from
        # the authoritative Panel Chart / Performance Data Summary
        # table first — this is what keeps narrative/partial-data
        # chunks from crowding out the real attribute table.
        docs = prioritize_structured_chunks(
            search_query,
            deduped_docs,
            top_k=rerank_top_k
        )
    else:
        docs = bm25_rerank(
            search_query,
            deduped_docs,
            top_k=rerank_top_k
        )
    # =================================================
    # CONTEXT BUILDING
    # =================================================
    # Stop adding chunks once the budget is reached rather than
    # slicing the final joined string — slicing mid-string could
    # cut a chunk (and a table row) in half; stopping at a whole
    # chunk boundary never does, and keeps the highest-priority
    # chunks (model-code matches, ordered by BM25 relevance) fully
    # intact.
    #
    # Groq's free/on_demand tier for openai/gpt-oss-20b caps total
    # tokens per request (prompt + completion) at 8000 TPM. This
    # app's rules/instructions text alone is ~1200 tokens, plus
    # ~150 for the system prompt, ~900 reserved for the answer
    # (see llm.py's max_tokens) and ~20 for the question. 9000
    # characters of context is roughly 2800-3000 tokens at this
    # model's tokenization of number/table-heavy text, keeping the
    # full request safely under the 8000 cap with headroom to
    # spare. Raising this risks the "Request too large" 413 error
    # — if you do raise it, lower llm.py's max_tokens to compensate.
    MAX_CONTEXT_CHARS = 9000
    context_parts = []
    seen = set()
    total_len = 0
    for doc in docs:
        chunk = doc.page_content.strip()
        if chunk in seen:
            continue
        seen.add(chunk)
        source = doc.metadata.get(
            "source",
            "Unknown"
        )
        page = doc.metadata.get(
            "page",
            "N/A"
        )
        piece = f"""DOCUMENT: {source}
PAGE: {page}
{chunk[:2200]}"""
        if total_len + len(piece) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(piece)
        total_len += len(piece)
    context = "\n\n".join(
        context_parts
    )
    # =================================================
    # PROMPT
    # =================================================
    prompt = f"""You are Tata Motors Department Knowledge Assistant.
Rules:
1. Use ONLY the context supplied.
2. Extract ALL numerical values if available.
3. If information exists in tables, include the values.
4. For Fuel/Fluid Economy questions:
   - Report EVERY (condition, speed) data point that actually
     appears in the context — e.g. Unladen/Laden/Overall at
     40 kmph and/or 55 kmph — as its OWN table row. Do NOT force
     the data into a fixed grid of all conditions × all speeds.
   - If a particular condition/speed combination is not present
     in the context, simply omit that row. NEVER invent a row,
     and NEVER fill a missing value with "N/A" or leave a cell
     blank — a row only exists if the value exists.
   - Order the rows: Unladen first, then Laden, then Overall;
     within each condition, 40 kmph before 55 kmph.
   - Copy every numeric value EXACTLY as given in the context
     (same digits, same decimal places, same unit). Do not round,
     truncate, or shorten a value (e.g. "4.20 kmpl" must stay
     "4.20 kmpl", never "4.2 km").
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
   a) SINGLE-SOURCE RULE (most important — prevents mixing data
      from different tables):
      - If more than one performance table for this vehicle
        appears in the context (e.g. a Panel Chart AND a separate
        "Performance Data Summary" / Annexure table), pick ONE
        table — the one with an explicit "Test Vehicle" (or
        "Current/Gateway" / "Status") column — and take EVERY
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
        or "Competitor" column values in the answer — those
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
      writing it — this is how to tell the columns apart even if
      headers are unclear or jumbled in the extracted text):
      - The Target / PALS value is a pass/fail THRESHOLD and is
        ALWAYS written with a "≤" or "≥" symbol (e.g. "≤66.30",
        "≥22.04") — never a bare number.
      - The actual measured Test Vehicle / Current-Gateway /
        Status result is ALWAYS a plain number with NO "≤" or "≥"
        symbol (e.g. "65.49", "22.56").
      - If you are about to put a value WITH a "≤"/"≥" symbol into
        a "Result" / "Test Vehicle" / "Current-Gateway" column or
        field, STOP — that value belongs in the Target column
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
        "Bathtub Driveability") table — this breaks each gear
        into its own speed bands (e.g. "3rd Gear (10-15 kmph)",
        "3rd Gear (15-20 kmph)", "4th Gear (20-25 kmph)", and so
        on). Report EVERY gear/speed-band row that exists in the
        context — this may run from 3rd Gear through 6th, 7th, or
        8th Gear depending on the document; never assume it stops
        at 6th, and never omit a row that is present.
      - Do NOT report the simple "Rate of Acceleration" table
        (one plain value per gear, e.g. "1st Gear: 2.26 kmph/s",
        no speed bands) UNLESS the Bath-Tub Drivability table is
        not present anywhere in the context for this vehicle —
        only then use Rate of Acceleration so gear data isn't
        omitted entirely from the answer.
      - Sustain Gradeability (%) and Restart Gradeability (%).
7. FORMATTING (very important):
   - Whenever the answer contains several data points that vary
     by condition, speed, gear, or test run, present them as a
     "tidy" Markdown table: one row per data point, with a
     column for each distinguishing factor plus a value column.
     Example shape (adapt the columns to whatever the context
     actually reports — do not force AC ON/OFF if that's not the
     variable here):
     | Condition | Speed  | Fuel Economy |
     |-----------|--------|--------------|
     | Unladen   | 55 kmph | 6.19 kmpl   |
     | Overall   | 40 kmph | 4.20 kmpl   |
     | Overall   | 55 kmph | 3.83 kmpl   |
   - Only include a row when every value in it is present in the
     context. Never pad a table with empty or "N/A" cells to make
     it look like a complete grid.
   - Always put a blank line before and after every table.
   - Use bullet points (not tables) for single, non-comparative
     values.
   - Keep normal sentence spacing; do not merge lines together.
8. Do NOT show:
   - PDF names
   - source names
   - page numbers
   - citations
9. Give concise engineering answers, but do not omit a data
   point just to keep the answer short — completeness matters
   more than brevity when reporting comparative or panel-chart
   data.
10. Reply with:
Information not found in documents.
Only if no relevant information exists.
CONTEXT:
{context}
QUESTION:
{question}"""
    # =================================================
    # LLM
    # =================================================
    try:
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
        # CLEANUP
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
        # down to a single plain "kmph" so nothing renders as a
        # broken superscript in the UI.
        answer = re.sub(
            r"km\s*/?\s*h(?:\s*(?:⁻\s*1|-\s*1|\^-1))?",
            "kmph",
            answer,
            flags=re.IGNORECASE
        )
        # NOTE: only collapse repeated SPACES/TABS, never newlines.
        # The old r"\s{2,}" pattern also matched newlines, which
        # squashed multi-line Markdown tables into one line and
        # broke their formatting.
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
        # ANSWER DISPLAY
        # =============================================
        st.subheader("Answer")
        st.markdown(answer)
    except Exception as e:
        st.error(
            f"LLM Error: {e}"
        )
    # =================================================
    # SOURCE DISPLAY
    # =================================================
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
    with st.expander(
        "📚 Retrieved Sources"
    ):
        for source, pages in source_pages.items():
            st.write(
                f"📄 {source} | Pages: {sorted(list(pages))}"
            )
    # =================================================
    # DEBUG
    # =================================================
    with st.expander(
        "⚙️ Debug Information",
        expanded=False
    ):
        st.write(
            f"Model codes detected: {model_codes}"
        )
        st.write(
            f"Exact model-code matches: {len(exact_docs)}"
        )
        st.write(
            f"Retrieved Chunks (final): {len(docs)}"
        )
        for i, doc in enumerate(
            docs,
            start=1
        ):
            st.markdown(
                f"### Chunk {i}"
            )
            st.write(
                doc.metadata
            )
            st.write(
                doc.page_content[:1200]
            )
