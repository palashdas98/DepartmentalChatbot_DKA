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
    if (
        "fuel economy" in lower_question
        or "fe" in lower_question
        or "mileage" in lower_question
    ):
        search_query += """
        fuel economy
        mileage
        FE
        kmpl
        km/l
        fuel consumption
        """
    elif (
        "acceleration" in lower_question
    ):
        search_query += """
        acceleration
        0-60 kmph
        0-1000 m
        WOT
        driveability
        """
    elif (
        "performance" in lower_question
    ):
        search_query += """
        acceleration
        fuel economy
        WOT
        driveability
        restart gradeability
        """
    # =================================================
    # RETRIEVAL
    # =================================================
    with st.spinner(
        "Searching Knowledge Base..."
    ):
        results = (
            vectorstore.similarity_search_with_score(
                search_query,
                k=20
            )
        )
    docs = [
        doc
        for doc, score in results
    ]
    docs = bm25_rerank(
        search_query,
        docs,
        top_k=5
    )
    # =================================================
    # CONTEXT BUILDING
    # =================================================
    context_parts = []
    seen = set()
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
        context_parts.append(
            f"""DOCUMENT: {source}
PAGE: {page}
{chunk[:1200]}"""
        )
    context = "\n\n".join(
        context_parts
    )
    # Protection from Groq token limits
    context = context[:6000]
    # =================================================
    # PROMPT
    # =================================================
    prompt = f"""You are Tata Motors Department Knowledge Assistant.
Rules:
1. Use ONLY the context supplied.
2. Extract ALL numerical values if available.
3. If information exists in tables, include the values.
4. For Fuel Economy questions, provide FE, mileage, kmpl, fuel consumption values.
5. For Performance questions, provide:
   - 0-60 kmph
   - 0-1000 m
   - WOT Driveability
   - Restart Gradeability
   - Acceleration
   - Sustainability
6. If gear-wise acceleration is present, format clearly:
   • 1st Gear : value
   • 2nd Gear : value
   • 3rd Gear : value
   • 4th Gear : value
   • 5th Gear : value
   • 6th Gear : value
7. FORMATTING (very important):
   - Whenever the answer contains values that vary by condition
     (e.g. AC ON/OFF, different speeds, different gears, different
     test runs), present them as a proper Markdown table with a
     header row, using this exact style:
     | Condition | 40 km/h | 55 km/h |
     |-----------|---------|---------|
     | AC OFF    | 3.60 kmpl | 3.40 kmpl |
     | AC ON     | 3.47 kmpl | 3.27 kmpl |
   - Always put a blank line before and after every table.
   - Use bullet points (not tables) for single, non-comparative
     values.
   - Keep normal sentence spacing; do not merge lines together.
8. Do NOT show:
   - PDF names
   - source names
   - page numbers
   - citations
9. Give concise engineering answers.
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
            f"Retrieved Chunks: {len(docs)}"
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