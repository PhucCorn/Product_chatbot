import re
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from splitter import *

# Văn bản mẫu
document_text = """Chương 1: TỔNG QUAN VỀ BAO BÌ
content
1.1 ĐẶC ĐIỂM CHUNG
content
1.1.1 Đặc điểm chung của bao bì
content
1.1.2 Tính năng của bao bì
content
1.1.2.1 Tính năng chứa đựng sản phẩm
content
1.2 ĐẶC ĐIỂM CHUNG
content
1.2.1 Đặc điểm chung của hộp giấy
content
"""

# Tạo tài liệu từ văn bản trên
doc = Document(page_content=document_text)

# Định nghĩa regex pattern cho các tiêu đề (sections)
patterns = [
    r'(?<=\n)Chương \d+: .+',  # Mục lớn như "Chương 1: TỔNG QUAN VỀ BAO BÌ"
    r'(?<=\n)\d+\.\d+ .+',  # Mục nhỏ hơn như "1.1 ĐẶC ĐIỂM CHUNG"
    r'(?<=\n)\d+\.\d+\.\d+ .+',  # Mục nhỏ hơn nữa như "1.1.1 Đặc điểm chung của bao bì"
    r'(?<=\n)\d+\.\d+\.\d+\.\d+ .+'  # Mục nhỏ hơn nữa như "1.1.2.1 Tính năng chứa đựng sản phẩm"
]

# Chia tài liệu thành các đoạn
loaders = TextLoader("doc/"+"spkt.txt", encoding = 'UTF-8')
docs = [loaders.load()]
splitter = PatternSplitter(patterns)
split_docs = splitter.split_documents(docs[0])
# In các đoạn
def beautiful_print(split_docs):
    for split_doc in split_docs:
        print("Chunk:")
        print(" ".join([split_doc.metadata['chapter'],split_doc.metadata['section'],split_doc.metadata['subsection'],split_doc.metadata['subsubsection'],]))
        print("----------")
        beautiful_print(split_doc.metadata["subtree"])

         
beautiful_print(split_docs)

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# loaders = TextLoader("doc/"+"spkt.txt", encoding = 'UTF-8')
# docs= [loaders.load()]
# # print(docs[0][0].page_content)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,  separators=["\n\n"])
# docs = text_splitter.split_documents(docs[0])
# print(docs[0])
