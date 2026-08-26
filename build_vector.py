import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

all_docs = []

pdf_folder = "pdfs"

print("Loading PDFs...")

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        filepath = os.path.join(pdf_folder, file)

        loader = PyPDFLoader(filepath)

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file

        all_docs.extend(docs)

print("Documents Loaded:", len(all_docs))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks = splitter.split_documents(all_docs)

print("Chunks Created:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

print("Vector Database Created Successfully")