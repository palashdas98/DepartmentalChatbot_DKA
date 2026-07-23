import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm import get_llm

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Department RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Department RAG Chatbot")

# =========================================
# LOAD VECTORSTORE
# =========================================

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

try:

    vectorstore = load_vectorstore()

    st.success("✅ Vector Database Loaded")

except Exception as e:

    st.error(f"Vector Database Load Error: {e}")

    st.stop()

# =========================================
# QUESTION
# =========================================

question = st.text_input(
    "Ask a question from your PDFs"
)

if question:

    with st.spinner("Searching Documents..."):

        docs = vectorstore.max_marginal_relevance_search(
            question,
            k=10,
            fetch_k=30
        )

    st.subheader("Debug Information")

    st.write(
        f"Retrieved Chunks: {len(docs)}"
    )

    if len(docs) == 0:

        st.error(
            "No matching chunks found."
        )

        st.stop()

    # =====================================
    # SHOW RETRIEVED CHUNKS
    # =====================================

    with st.expander("Retrieved Chunks"):

        for i, doc in enumerate(docs):

            st.markdown(
                f"### Chunk {i+1}"
            )

            st.write(
                doc.metadata
            )

            st.write(
                doc.page_content
            )

            st.divider()

    # =====================================
    # BUILD CONTEXT
    # =====================================

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    st.write(
        "Context Length:",
        len(context)
    )

    with st.expander(
        "Context Sent To LLM"
    ):

        st.write(context)

    # =====================================
    # PROMPT
    # =====================================

    prompt = f"""
You are a Tata Motors departmental assistant.

Answer ONLY from the context.

Rules:

1. Use information from all chunks.
2. Combine information from multiple chunks if required.
3. If numerical values exist, report them exactly.
4. Do NOT invent information.
5. If answer is unavailable, say:
   "Information not found in documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # =====================================
    # GROQ
    # =====================================

    try:

        llm = get_llm()

        answer = llm.invoke(prompt)

        st.subheader(
            "Raw LLM Response"
        )

        st.write(answer)

        st.subheader("Answer")

        st.success(answer)

    except Exception as e:

        st.error(
            f"LLM Error: {e}"
        )

    # =====================================
    # SOURCE DOCUMENTS
    # =====================================

    st.subheader(
        "Source Documents"
    )

    sources = set()

    for doc in docs:

        if "source" in doc.metadata:

            sources.add(
                doc.metadata["source"]
            )

    for src in sources:

        st.write(
            f"📄 {src}"
        )