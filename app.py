import streamlit as st
from collections import defaultdict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm import get_llm

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Department Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Department Knowledge Assistant")

# =====================================================
# LOAD VECTOR DATABASE
# =====================================================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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
# QUESTION
# =====================================================

question = st.text_input(
    "Ask a question from your PDFs"
)

if question:

    # =================================================
    # QUERY EXPANSION
    # =================================================

    expanded_query = f"""
Question: {question}

Related Terms:
Acceleration
0-60 kmph
0-60 km/h
Performance
Vehicle Performance
Test Results
Speed
Acceleration Time
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
"""
    
    # =================================================
    # RETRIEVAL
    # =================================================

    with st.spinner("Searching Knowledge Base..."):

        docs = vectorstore.similarity_search(
            expanded_query,
            k=15,
            fetch_k=50
        )

    st.subheader("Debug Information")

    st.write(
        f"Retrieved Chunks: {len(docs)}"
    )

    if len(docs) == 0:

        st.error(
            "No matching documents found."
        )

        st.stop()

    # =================================================
    # SHOW RETRIEVED CHUNKS
    # =================================================

    with st.expander("Retrieved Chunks"):

        for i, doc in enumerate(docs):

            st.markdown(
                f"### Chunk {i+1}"
            )

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "N/A"
            )

            st.write(
                f"📄 Source: {source}"
            )

            st.write(
                f"📖 Page: {page}"
            )

            st.write(
                doc.page_content
            )

            st.divider()

    # =================================================
    # BUILD CONTEXT
    # =================================================

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

        chunk_text = doc.page_content

        context += f"""
SOURCE DOCUMENT: {source}
PAGE NUMBER: {page}

{chunk_text}

======================================================
"""

    st.write(
        "Context Length:",
        len(context)
    )

    with st.expander(
        "Context Sent To LLM"
    ):
        st.text(context)

    # =================================================
    # PRIMARY SOURCE
    # =================================================

    top_doc = docs[0]

    primary_source = top_doc.metadata.get(
        "source",
        "Unknown"
    )

    primary_page = top_doc.metadata.get(
        "page",
        "N/A"
    )

    # =================================================
    # PROMPT
    # =================================================

    prompt = f"""
You are a Tata Motors departmental knowledge assistant.

Instructions:

1. Use ONLY the supplied context.
2. Never invent information.
3. Combine information from multiple chunks when required.
4. Mention exact values and units.
5. Mention exact limits and specifications.
6. If answer exists partially, provide the available information.
7. Use bullet points whenever possible.
8. At the end mention source document and page number.
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
<source file>

Page:
<page number>
"""

    # =================================================
    # CALL GROQ
    # =================================================

    try:

        llm = get_llm()

        answer = llm.invoke(prompt)

        st.subheader("Answer")

        st.success(answer)

    except Exception as e:

        st.error(
            f"LLM Error: {e}"
        )

        st.stop()

    # =================================================
    # PRIMARY SOURCE
    # =================================================

    st.subheader("Primary Source")

    st.info(
        f"📄 {primary_source} | 📖 Page {primary_page}"
    )

    # =================================================
    # ALL RETRIEVED SOURCES
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

    st.subheader("Retrieved Sources")

    for source, pages in source_pages.items():

        page_list = sorted(
            list(pages)
        )

        st.write(
            f"📄 {source} | Pages: {page_list}"
        )