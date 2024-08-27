from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import pandas as pd
from tabulate import tabulate


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

def is_relevance(sub_text, main_text):
    tagging_prompt = ChatPromptTemplate.from_template(
        """
    Evaluate the relevance of the following image caption to the provided text.

    Only extract the properties mentioned in the 'Classification' function.

    Image caption:
    {sub_text}
    Text:
    {main_text}
    """
    )
    class Classification(BaseModel):
        relevance: bool = Field(description="How relevance of the following image caption to the provided text. The relevance is just True or False, where True means image caption and the provided text are revelant, and False means not revelant")
        relevance_score : float = Field(
            description="How relevance of the following image caption to the provided text. The relevance is on a scale of 1 to 10, where 1 means completely unrelated, and 10 means highly relevant."
        )
        explanation : str = Field(description="a brief explanation for your relevance score")
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0).with_structured_output(
        Classification
    )   
    tagging_chain = tagging_prompt | llm
    res = tagging_chain.invoke({"sub_text": sub_text, "main_text": main_text})
    print(res.dict())
    return res.dict()["relevance"]

def tabel_gen(file_path):
    df_dict = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    tables = []
    sheets_name = []
    for sheet_name, df in df_dict.items():
        # table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        table = df.to_string(index=False)
        tables += [table]
        sheets_name += [sheet_name]
    return tables, sheets_name

def add_table(txt_file_path):
    xlsx_file_path = txt_file_path[:-4]+".xlsx"
    tables, sheets_name = tabel_gen(xlsx_file_path)
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    for i, sheet_name in enumerate(sheets_name):
        updated_content = content.replace("--xlsx."+sheet_name+"--", tables[i])
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)