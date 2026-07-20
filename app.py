import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm import ask_gpt

st.set_page_config(
    page_title="Department RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Department RAG Chatbot")

st.write(
    "Ask questions from your maintenance manuals, SOPs and department documents."
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

question = st.text_input(
    "Ask Question"
)

if question:

    docs = db.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    with st.spinner("Searching documents..."):

        answer = ask_gpt(
            question,
            context
        )

    st.subheader("✅ Answer")

    st.write(answer)

    with st.expander("Retrieved Context"):

        st.write(context[:5000])