from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import OpenAIEmbeddings
import uuid
from dotenv import load_dotenv
import time
import shutil
from langchain.retrievers.multi_vector import SearchType

import os

thư_mục_chroma = "./chroma_langchain_db"  # Thay thế bằng đường dẫn thư mục Chroma của bạn

if os.path.exists(thư_mục_chroma):
    shutil.rmtree(thư_mục_chroma)
    print("Đã xóa thư mục Chroma cũ.")

load_dotenv()
loaders = [
    TextLoader("doc\spkt.txt", encoding = 'UTF-8'),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
docs = text_splitter.split_documents(docs)
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"), persist_directory="./chroma_langchain_db",
)
store = InMemoryByteStore()
id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
retriever.search_type = SearchType.mmr
with open('img_cap.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
result = retriever.invoke(lines[0][2:]+" và "+lines[1][2:])
for doc in result:
    print(doc.page_content)
    print("------------------------------------------------------------")
