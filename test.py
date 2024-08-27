from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
loaders = [
    TextLoader("doc/"+'spkt.txt', encoding = 'UTF-8'),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
docs = text_splitter.split_documents(docs)
for doc in docs:
    print(doc.page_content)
    print("////////////////////////////////////////////")