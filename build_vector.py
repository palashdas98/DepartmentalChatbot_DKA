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

print("\nLoading PDFs...\n")

for file in os.listdir(pdf_folder):

    if file.lower().endswith(".pdf"):

        print(f"Loading PDF: {file}")

        path = os.path.join(
            pdf_folder,
            file
        )

        loader = PyPDFLoader(path)

        docs = loader.load()

        for doc in docs:

            doc.metadata["source"] = file

        all_docs.extend(docs)

print(
    f"\nDocuments Loaded: {len(all_docs)}"
)

# Chunk size raised from 1500 -> 2200 (overlap raised to match)
# so that a multi-row fuel-economy table (Unladen/Laden/Overall
# x 40/55 kmph) is much less likely to get split mid-table across
# two separate chunks. A split table means retrieval can pull in
# only half the rows even when top_k is generous.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2200,
    chunk_overlap=400
)

chunks = splitter.split_documents(
    all_docs
)

print(
    f"Chunks Created: {len(chunks)}"
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

db = FAISS.from_documents(
    chunks,
    embeddings
)

db.save_local(
    "vectorstore"
)

print(
    "\n✅ Vectorstore Created Successfully"
)