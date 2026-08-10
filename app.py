import streamlit as st
from collections import defaultdict
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm import get_llm

st.set_page_config(
    page_title="Department Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Department Knowledge Assistant")


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


question = st.text_input(
    "Ask a question from your PDFs"
)

if question:

    with st.spinner(
        "Searching Knowledge Base..."
    ):

        results = vectorstore.similarity_search_with_score(
            question,
            k=4
        )

    docs = [doc for doc, score in results]

    # ==========================
    # DEBUG SECTION (HIDDEN)
    # ==========================

    with st.expander("⚙️ Debug Information", expanded=False):

        st.write(
            f"Retrieved Chunks: {len(docs)}"
        )

        with st.expander("Retrieved Chunks"):

            for i, (doc, score) in enumerate(results):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.write(
                    f"Score: {round(score,4)}"
                )

                st.write(
                    f"Source: {doc.metadata.get('source')}"
                )

                st.write(
                    f"Page: {doc.metadata.get('page')}"
                )

                st.write(doc.page_content)

                st.divider()

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

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a Tata Motors Department Knowledge Assistant.

Answer only from the provided context.

Give a concise and direct answer.

DO NOT mention:
- Source file names
- PDF names
- Page numbers
- References

If information is not available, reply exactly:

Information not found in documents.

CONTEXT:

{context}

QUESTION:

{question}
"""

    try:

        llm = get_llm()

        answer = llm.invoke(prompt)

        # Remove any source references if LLM still adds them
        answer = re.sub(
            r"\(Source:.*?\)",
            "",
            answer,
            flags=re.IGNORECASE
        ).strip()

        st.subheader("Answer")

        st.success(answer)

    except Exception as e:

        st.error(
            f"LLM Error: {e}"
        )

        st.stop()

    # ==========================
    # SOURCES DROPDOWN
    # ==========================

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
        "📚 Retrieved Sources (Click to Expand)",
        expanded=False
    ):

        for source, pages in source_pages.items():

            st.write(
                f"📄 {source} | Pages: {sorted(list(pages))}"
            )