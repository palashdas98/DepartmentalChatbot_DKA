import os

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

all_docs = []

pdf_folder = "pdfs"

print("Loading PDFs...")

for file in os.listdir(pdf_folder):

    if file.endswith(".pdf"):

        filepath = os.path.join(pdf_folder, file)

<<<<<<< HEAD
        path = os.path.join(
            pdf_folder,
            file
        )

        loader = PyPDFLoader(path)
=======
        loader = PyPDFLoader(filepath)
>>>>>>> 5943ed513b498b440d438f7a2c7498d1333299cd

        docs = loader.load()

        for doc in docs:

            doc.metadata["source"] = file

        all_docs.extend(docs)

<<<<<<< HEAD
print(
    f"\nDocuments Loaded: {len(all_docs)}"
=======
print("Documents Loaded:", len(all_docs))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
>>>>>>> 5943ed513b498b440d438f7a2c7498d1333299cd
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

<<<<<<< HEAD
chunks = splitter.split_documents(
    all_docs
)

print(
    f"Chunks Created: {len(chunks)}"
)
=======
print("Chunks Created:", len(chunks))
>>>>>>> 5943ed513b498b440d438f7a2c7498d1333299cd

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

<<<<<<< HEAD
db.save_local(
    "vectorstore"
)

print(
    "\n✅ Vectorstore Created Successfully"
)
=======
print("Vector Database Created Successfully")
>>>>>>> 5943ed513b498b440d438f7a2c7498d1333299cd
