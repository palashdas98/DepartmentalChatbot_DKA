from collections import defaultdict

from fastapi import FastAPI
from fastapi import HTTPException

from pydantic import BaseModel

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from llm import get_llm
from hybrid_retriever import bm25_rerank


app = FastAPI(
    title="Department Knowledge Assistant",
    version="2.0"
)


class QuestionRequest(BaseModel):
    question: str


print("Loading Embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

print("Loading Vector Store...")

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Vector Store Loaded")


@app.get("/")
def home():

    return {
        "status": "running",
        "application":
        "Department Knowledge Assistant"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat(request: QuestionRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

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

    if not docs:

        return {
            "question": question,
            "answer":
            "Information not found in documents."
        }

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    prompt = f"""
Answer ONLY from the supplied context.

If answer does not exist reply exactly:

Information not found in documents.

CONTEXT:

{context}

QUESTION:

{question}
"""

    llm = get_llm()

    answer = llm.invoke(
        prompt
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

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "source": source,
                "pages": sorted(list(pages))
            }
            for source, pages
            in source_pages.items()
        ]
    }