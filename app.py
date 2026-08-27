import re
import streamlit as st

from collections import defaultdict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm import get_llm
from hybrid_retriever import bm25_rerank


# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Department Knowledge Assistant",
    page_icon="🚛",
    layout="wide"
)

st.title("Department Knowledge Assistant")
st.caption(
    "AI-powered document retrieval and question answering system"
)


# -------------------------------------------------
# LOAD VECTOR DATABASE
# -------------------------------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )


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


# -------------------------------------------------
# USER QUESTION
# -------------------------------------------------

question = st.text_input(
    "Ask a question from your PDFs"
)


# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------

if question:

    with st.spinner(
        "Searching Knowledge Base..."
    ):

        results = (
            vectorstore.similarity_search_with_score(
                question,
                k=15
            )
        )

    docs = [
        doc
        for doc, score in results
    ]

    docs = bm25_rerank(
        question,
        docs,
        top_k=10
    )

    # ---------------------------------------------
    # REMOVE DUPLICATE CHUNKS
    # ---------------------------------------------

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
            f"""
SOURCE: {source}
PAGE: {page}

{chunk}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # ---------------------------------------------
    # PERFORMANCE QUERY DETECTION
    # ---------------------------------------------

    performance_keywords = [
        "performance",
        "performance summary",
        "vehicle performance",
        "acceleration",
        "wot",
        "wide open throttle",
        "driveability",
        "drivability",
        "gradeability",
        "restart gradeability",
        "bath Tub",
        "bathtub"
    ]

    is_performance_query = any(
        keyword in question.lower()
        for keyword in performance_keywords
    )

    # ---------------------------------------------
    # PROMPT
    # ---------------------------------------------

    prompt = f"""
You are Tata Motors Department Knowledge Assistant.

Instructions:

1. Use the supplied context to answer the question.

2. If relevant information exists in the context, summarize it clearly.

3. Provide professional engineering answers.

4. Never show:
   - PDF names
   - file names
   - source names
   - page numbers
   - citations
   - chunk ids

5. Never generate HTML tags.

6. Never use:
   - km/h·s⁻¹
   - km/h/s
   - km/h-sec
   - km h⁻¹ s⁻¹

7. If exact value is unavailable but related performance information exists, provide it.

8. Only reply: Information not found in documents. when no relevant information exists.

9. Present numerical information in clean markdown tables whenever possible.

"""

    if is_performance_query:

        prompt += """

IMPORTANT:

User is asking vehicle performance information.

You MUST extract ALL available performance information present in context.

Provide these sections separately with clear headings:

### Acceleration

### Wide Open Throttle (WOT) Drivability

### Bat-Tub Driveability 

### Sustain & Restart Gradeability

Rules:

1. Never use bullet points.

2. Never skip an available section.

3. If a section is not available write:

Information not found in documents.

4. For WOT Drivability use format:

| Gear | Acceleration (Kmph/Sec) |
|------|------|

5. For Acceleration use format:

| Parameter | Value |

6. For Bat-Tub Driveability use format:

| Parameter | Value |

7. For Sustain & Restart Gradeability use format:

| Parameter | Value |

8. Compare vehicles only if comparison data exists.

9. Show every numerical value found in context.
"""

    prompt += f"""

CONTEXT:

{context}

QUESTION:

{question}
"""

    try:

        llm = get_llm()

        answer = llm.invoke(
            prompt
        )
        if not answer or not answer.strip():
            answer = (
                "Relevant documents were retrieved, "
                "but no answer was generated. "
                "Please refine the question."
            )
        # -----------------------------------------
        # CLEANING
        # -----------------------------------------

        answer = re.sub(
            r"<strong.*?>",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"</strong>",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"<[^>]+>",
            "",
            answer
        )

        answer = re.sub(
            r"&lt;br\s*/?&gt;",
            "\n",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"km/h·s⁻¹",
            "Kmph/Sec",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"km/h/s",
            "Kmph/Sec",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"km/h-sec",
            "Kmph/Sec",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"source\s*page\s*\d+",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"\n{3,}",
            "\n\n",
            answer
        )

        answer = answer.strip()

        # -----------------------------------------
        # DISPLAY ANSWER
        # -----------------------------------------

        st.subheader("Answer")

        st.markdown(
            answer
        )

    except Exception as e:

        st.error(
            f"LLM Error: {e}"
        )

    # ---------------------------------------------
    # SOURCE DISPLAY
    # ---------------------------------------------

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

    # ---------------------------------------------
    # DEBUG (OPTIONAL)
    # ---------------------------------------------

    # with st.expander("⚙️ Debug Information"):
    #
    #     st.write(
    #         f"Retrieved Chunks: {len(docs)}"
    #     )
    #
    #     for i, doc in enumerate(docs, start=1):
    #
    #         st.markdown(
    #             f"### Chunk {i}"
    #         )
    #
    #         st.write(
    #             doc.metadata
    #         )
    #
    #         st.write(
    #             doc.page_content[:1000]
    #         )