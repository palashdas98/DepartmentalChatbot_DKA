import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from llm import get_llm

st.set_page_config(
    page_title="Department Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Department Knowledge Assistant")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vectordb = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

st.success("✅ Vector Database Loaded Successfully")

question = st.text_input(
    "Ask a question from your PDFs"
)

if question:

    docs = vectordb.similarity_search(
        question,
        k=15
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = get_llm()

    answer = llm.invoke(
        context=context,
        question=question
    )

    st.subheader("Answer")

    st.success(answer)

    with st.expander("Retrieved Sources"):

        for i, doc in enumerate(docs, start=1):

            st.write(
                f"Source: "
                f"{doc.metadata.get('source')}"
            )

            st.info(
                doc.page_content[:1000]
            )