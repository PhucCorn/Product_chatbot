from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
# import pandas as pd
# from tabulate import tabulate


def get_session_history_mongodb(session_id):
    return MongoDBChatMessageHistory(
        'mongodb://localhost:27017/', 
        session_id, 
        database_name="product_chat_database", 
        collection_name="messages"
    )
    
def bbas_docs():
    docs = docs = [
        Document(
            page_content="Tuyên ngôn doanh nghiệp: \"là tuyên bố chính thức của doanh nghiệp đối với khách hàng, đối tác, người lao động, cộng đồng và các bên liên quan khác về doanh nghiệp.\"",
            metadata={"topic": "Giới thiệu chung về bao bì, phát triển hệ thống bao bì cho sản phẩm"},
        ),
        
    ]
    return docs

def is_relevance(img_caps, answer, message, relevant_docs):
    batch = []
    tagging_prompt = ChatPromptTemplate.from_template(
        """
    You are given four pieces of information: (1) a question, (2) an answer to the question, (3) a caption for an image that is intended to illustrate the answer, and (4) a document that is used to generate an answer and an image represents. Your task is to assess how relevant the image caption is to the given answer, considering that the image is used to support or clarify the answer. 

    Only extract the properties mentioned in the 'Classification' function.

    A question:
    {question}
    An answer:
    {answer}
    A caption for an image:
    {img_cap}
    A document:
    {relevant_docs}
    """
    )
    class Classification(BaseModel):
        relevance: bool = Field(description="How relevance of caption for an image to an answer base on question, an answer, a document and a caption of image. The relevance is just True or False, where True means a caption for an image and a question, an answer are relevant, and False means not relevant")
        relevance_score : float = Field(
            description="How relevance of the following image caption to a question and an answer. The relevance is a real number and is on a scale of 0 to 10, where 0 means completely irrelevant, and 10 means the caption perfectly complements and supports the answer."
        )
        explanation : str = Field(description="a brief explanation for your relevance score")
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0).with_structured_output(
        Classification
    )   
    tagging_chain = tagging_prompt | llm
    for img_cap in img_caps:
        batch += [{"question": message, "answer": answer, "img_cap": img_cap, "relevant_docs": relevant_docs}]
    response = tagging_chain.batch(batch)
    return response

def img_caps_gen(docs):
    img_caps = []
    for doc in docs:
        for img_cap in doc.metadata["img_cap"]:
            if img_cap not in img_caps:
                img_caps += [img_cap]
    return img_caps

def summary_chain():
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("""
        You are an AI assistant tasked with summarizing a document to enhance its relevance for query-based retrieval. Please create a summary that:
        Focuses on the most relevant information that would likely be sought in a query.
        Prioritizes key terms, topics, and concepts that are central to understanding the document's content.
        Excludes any superfluous details, repetitive information, or sections not essential to common queries.
        Maintains the context necessary for accurate and useful responses during future searches.
        Document to summarize:  \n\n{doc}
        """)
        | ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
        | StrOutputParser()
    )
    return chain

def content_finder(node):
    if node.metadata["subtree"] == []:
        if node.metadata["rank"] == 2:
            result = " ".join([node.metadata["section"], node.page_content])
        elif node.metadata["rank"] == 3:
            result = " ".join([node.metadata["subsection"], node.page_content])
        elif node.metadata["rank"] == 4:
            result = " ".join([node.metadata["subsubsection"], node.page_content])
        return [result]
    else:
        if node.metadata["rank"] == 1:
            contents = [node.metadata["chapter"]]
        elif node.metadata["rank"] == 2:
            contents = [node.metadata["section"]]
        elif node.metadata["rank"] == 3:
            contents = [node.metadata["subsection"]]
        for sub_node in node.metadata["subtree"]:
            contents += content_finder(sub_node)
        return contents

