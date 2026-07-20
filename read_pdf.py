
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdfs/BS6_Manual.pdf")

pages = loader.load()

print(pages[0].page_content)
