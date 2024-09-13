from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import (
    trim_messages,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from pymongo import MongoClient
from datetime import datetime
from operator import itemgetter
import json
import ast


def get_session_history_mongodb(session_id):
    return MongoDBChatMessageHistory(
        'mongodb://localhost:27017/', 
        session_id, 
        database_name="product_chat_database", 
        collection_name="messages"
    )
    
def delete_two_most_recent_message(session_id):
    mongo_uri = 'mongodb://localhost:27017/'
    database_name = 'product_chat_database'
    collection_name = 'messages'
    client = MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]
    messages = list(collection.find({'SessionId': 'phuccngo'},sort=[('timestamp', -1)]))
    for _ in range(2):
        try:
            # Retrieve the most recent message
            most_recent_message = list(collection.find({'SessionId': 'phuccngo'}))[-1]
            if most_recent_message:
                most_recent_message_id = most_recent_message['_id']
                # message = json.loads(most_recent_message['History'])['data']['content']
                # Delete the most recent message
                collection.delete_one({'_id': most_recent_message_id})
                # print(f"Deleted message with message: {message}")
            else:
                print("No messages found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    # Close the MongoDB connection
    client.close()
    
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
        You are an AI assistant tasked with summarizing a document to enhance its relevance for query-based retrieval. Please create a summary in English that:
        - Focuses on the most relevant information that would likely be sought in a query.
        - Prioritizes key terms, topics, and concepts that are central to understanding the document's content.
        - Excludes any superfluous details, repetitive information, or sections not essential to common queries.
        - Maintains the original document's section headings and numbering (e.g., "2.1.3.1 Characteristics of the product") to ensure that the summary reflects the structure and context of the original document.
        - Provides context necessary for accurate and useful responses during future searches.
        In language: English
        Document to summarize:  

        {doc}
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
    
def vn_2_en(text):
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
    template = """
    Translate the following Vietnamese phrases to English:

    {phrases}

    Only translate the phrases and retain any specific names like 'Bao Bì Ánh Sáng' or 'BBAS'.
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    result = chain.invoke({"phrases": text})
    return result.content

def connect_database():
    username = "postgres"
    password = "12345678"
    host = "localhost"
    port = "5432"
    database = "postgres"
    postgres_uri = f"postgresql://{username}:{password}@{host}:{port}/{database}"
    db = SQLDatabase.from_uri(postgres_uri)
    return db

def routing(question, sql_info, config):
    class SQLClassification(BaseModel):
        is_sql_related: bool = Field(description="True if the question is related to SQL, otherwise False")
    template = """
    You are a helpful assistant with expertise in SQL and database management. Your task is to determine if the provided question is related to querying a SQL database.
    
    A SQL database infor: \n{sql_info}\n
    A message history: \n{history}\n
    Question: "{question}"

    Please classify whether the question is SQL-related or not, and provide a brief reason.

    Response format:
    is_sql_related: <True/False>
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0).with_structured_output(SQLClassification)
    token_counter_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    trimmer = trim_messages(
        max_tokens=500,
        strategy="last",
        token_counter=token_counter_model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    chain = (
        RunnablePassthrough.assign(messages=itemgetter("question") | trimmer)
        | (lambda output: (output.update({'question': output['messages']}), output)[-1])
        | (lambda output: (
            output.update({'history': output['question'][:-1]}),
            output.update({'question': output['question'][-1]}),
            output)[-1])
        | prompt
        | llm
        | (lambda output: str(output.is_sql_related))
    )
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history_mongodb,
        input_messages_key="question", #Which key is user's input
    )
    response = with_message_history.invoke(
        {
            "sql_info": sql_info, 
            'question': question
        },
        config=config,
    )
    delete_two_most_recent_message(config['configurable']['session_id'])
    return response

def get_table_info_with_full_packagings_samples(db):
    packagings_info = db.get_table_info(['packagings'])
    packagings_samples = db.run('SELECT * FROM packagings;')
    packagings_samples = ast.literal_eval(packagings_samples)
    samples = ""
    for line in packagings_samples[3:]:
        samples += "\t".join(line)+'\n'

    sql_info = db.get_table_info([item for item in db.get_table_names() if item != 'packagings'])
    return packagings_info[:-2]+samples+"\n"+sql_info




