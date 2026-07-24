import streamlit as st
from collections import Counter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm import get_llm

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Department RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Department RAG Chatbot")

# =====================================================
# LOAD VECTOR DATABASE
# =====================================================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

try:

    vectorstore = load_vectorstore()

    st.success("✅ Vector Database Loaded Successfully")

except Exception as e:

    st.error(f"Vector DB Load Error: {e}")
    st.stop()

# =====================================================
# QUESTION INPUT
# =====================================================

question = st.text_input(
    "Ask a question from your PDFs"
)

# =====================================================
# PROCESS QUESTION
# =====================================================

if question:

    search_query = f"""
Question: {question}

Related Terms:
Fuel Economy
FE
FLE
Mileage
Fuel Consumption
Road Speed
Maximum Speed
Vehicle Speed
Speed Governor
Performance
BSFC
Fuel Efficiency
"""

    with st.spinner("Searching Documents..."):

        docs = vectorstore.max_marginal_relevance_search(
            search_query,
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
    # RETRIEVED CHUNKS
    # =================================================

    with st.expander("Retrieved Chunks"):

        for i, doc in enumerate(docs):

            st.markdown(
                f"### Chunk {i+1}"
            )

            st.write(
                f"Source: {doc.metadata.get('source','Unknown')}"
            )

            st.write(doc.page_content)

            st.divider()

    # =================================================
    # CONTEXT BUILDING
    # =================================================

    context = ""

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        context += (
            f"\nSOURCE FILE: {source}\n"
        )

        context += doc.page_content

        context += "\n\n"

    st.write(
        "Context Length:",
        len(context)
    )

    with st.expander(
        "Context Sent To LLM"
    ):
        st.write(context)

    # =================================================
    # PROMPT
    # =================================================

    prompt = f"""
You are a Tata Motors Departmental Knowledge Assistant.

Instructions:

1. Use ONLY information from the supplied context.
2. Combine information from multiple chunks.
3. Mention important numbers exactly.
4. Mention limits exactly.
5. Mention units exactly.
6. If information exists partially, provide the available information.
7. Do NOT invent information.
8. Mention the most relevant source document.
9. Use bullet points whenever possible.
10. Only say 'Information not found in documents'
    when absolutely nothing relevant exists.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # =================================================
    # GROQ LLM
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

    # =================================================
    # SOURCE DOCUMENTS
    # =================================================

    st.subheader("Source Documents")

    source_counter = Counter()

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        source_counter[source] += 1

    best_source = max(
        source_counter,
        key=source_counter.get
    )

    st.info(
        f"Most Relevant Source: {best_source}"
    )

    for source, count in source_counter.items():

        st.write(
            f"📄 {source}  (Relevant Chunks: {count})"
        )