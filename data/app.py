import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(
    page_title="Department RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Department RAG Chatbot")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

question = st.text_input(
    "Ask a question from your PDF"
)

if question:

    docs = vectordb.similarity_search(
        question,
        k=3
    )

    st.subheader("Answer")

    if len(docs) > 0:
        answer = docs[0].page_content
        st.success(answer)

        st.subheader("Source Chunks")

        for i, doc in enumerate(docs, start=1):
            st.write(f"Chunk {i}")
            st.info(doc.page_content)

    else:
        st.error("No answer found.")