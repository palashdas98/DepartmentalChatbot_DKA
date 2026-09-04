from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# IMPORTANT: this must match the model used in build_vector.py
# and app.py/api.py. It was previously
# "sentence-transformers/all-mpnet-base-v2", which does NOT match
# the model the vectorstore was actually built with — that mismatch
# means every query embedding here was computed in a different
# vector space than the index, making similarity scores meaningless.
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "40 kmph"

results = db.similarity_search(
    query,
    k=10
)

print(f"Documents Found: {len(results)}")

for i, doc in enumerate(results):

    print("\n")
    print("=" * 50)

    print(f"Document {i+1}")

    print(doc.metadata)

    print(doc.page_content[:1000])
