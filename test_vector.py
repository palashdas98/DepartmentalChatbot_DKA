from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "40 kmph"

results = db.similarity_search(
    query,
    k=5
)

print(f"Documents Found: {len(results)}")

for i, doc in enumerate(results):

    print("\n")
    print("=" * 50)

    print(f"Document {i+1}")

    print(doc.metadata)

    print(doc.page_content[:1000])