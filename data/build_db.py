from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("Loading PDF...")

loader = PyPDFLoader("pdfs/Manual.pdf")
documents = loader.load()

print(f"Pages Loaded: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Chunks Created: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

print("Database Created Successfully")