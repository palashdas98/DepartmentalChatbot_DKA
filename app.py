import re

import streamlit as st

from collections import defaultdict

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from llm import get_llm
from hybrid_retriever import bm25_rerank


st.set_page_config(
    page_title="Department Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title(
    "🤖 Department Knowledge Assistant"
)


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


question = st.text_input(
    "Ask a question from your PDFs"
)

if question:

    with st.spinner(
        "Searching Knowledge Base..."
    ):

        results = (
            vectorstore
            .similarity_search_with_score(
                question,
                k=20
            )
        )

    docs = [
        doc
        for doc, score in results
    ]

    docs = bm25_rerank(
        question,
        docs,
        top_k=8
    )

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

    prompt = f"""
Answer ONLY using the supplied context.

Provide concise and accurate answers.

If answer is unavailable reply exactly:

Information not found in documents.

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

        answer = re.sub(
            r"Source:.*",
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

    with st.expander(
        "⚙️ Debug Information"
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
                doc.page_content[:1000]
            )