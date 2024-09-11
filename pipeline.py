from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    trim_messages,
)
from langchain.storage import InMemoryByteStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from sqlalchemy.exc import ProgrammingError
import uuid
import shutil
from util import *
import os
from retriever import *
from splitter import *

class AIAssistant:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        self.parser = StrOutputParser()
        # self.retriever = self.tree_retriever()
        
    def llm_memory(self):
        trimmer = trim_messages(
            max_tokens=500,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        query = {'username': 'phuccngo'}
        dbclient = MongoClient('mongodb://localhost:27017/')
        db = dbclient['product_chat_database']
        conversations_collection = db['messages']
        result = conversations_collection.find_one(query)
        return trimmer
    
    def docs_prompt_gen(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Bạn là AI Chatbot của Công ty CP TM & SX Bao Bì Ánh Sáng hay BBAS. 
                    Bạn ở đây để trả lời tất cả các câu hỏi về các sản phẩm của doanh nghiệp với các tài liệu bạn được cung cấp. 
                    Tài liệu được cung cấp: {doc} 
                    Nếu tài liệu được cung cấp là "Không có tài liệu liên quan đến câu hỏi hoặc yêu cầu này." hoặc các tài liệu được cung cấp không liên quan đến câu hỏi hoặc yêu cầu thì trả lời là \"Tôi không được cung cấp thông tin để trả lời câu hỏi này\"
                    Lưu ý: CÁC CHỈ TRẢ LỜI CÁC CÂU HỎI HOẶC YÊU CẦU ĐƯỢC HỎI, KHÔNG THỪA, KHÔNG THIẾU VÀ KHÔNG TỰ Ý TÓM TẮT HAY RÚT NGẮN CÂU TRẢ LỜI
                    Câu hỏi:
                    """,
                ),
                MessagesPlaceholder(variable_name="question"), #phần "question" bên dưới sẽ được chèn vào đây
            ]
        )
        return prompt
    
    def sql_prompt_gen(self):
        prompt = PromptTemplate.from_template(
                    """
                    Bạn là AI Chatbot của Công ty CP TM & SX Bao Bì Ánh Sáng hay BBAS. 
                    Bạn ở đây để trả lời tất cả các câu hỏi về các sản phẩm của doanh nghiệp với các thông tin bạn được cung cấp. 
                    Bạn sẽ được cung cấp câu hỏi của người dùng, SQL query tương ứng, và kết quả truy xuất SQL, trả lời câu hỏi của người dùng.
                    Câu hỏi: {question}
                    SQL Query: {query}
                    Kết quả SQL: {result}
                    Nếu không có thông tin để trả lời, câu hỏi không liên quan, hoặc câu trả lời không liên quan đến câu hỏi, hoặc SQL Query bị lỗi thì trả lời là \"Không có dữ liệu để truy vấn\"
                    Nhớ ghi lại các chú thích ảnh trong câu trả lời.
                    Câu trả lời: 
                    """
        )
        return prompt
    
    def sql_code_prompt_gen(self, history):
        examples = [
            {
                "input": "kích thước túi pe cả ảnh minh họa", 
                "query": """SELECT "size_type", "length", "tolerance", "image_path" 
                            FROM "packaging_sizes"
                            WHERE "id" = 'BT1KH00051KP0001'
                            AND "size_type" = 'Túi PE'
                            LIMIT 5;"""
            },
            {
                "input": "chiều ngang bao của Bao dán đáy 3 lớp giấy có cả ảnh minh họa", 
                "query": """SELECT "size_type", "length", "tolerance", "image_path" 
                            FROM "packaging_sizes"
                            WHERE "id" = 'BT0KH00000KP0000'
                            AND "size_type" = 'Chiều ngang bao'
                            LIMIT 5;"""
            },
            {
                "input": "chiều rộng đáy bao của Bao dán đáy 3 lớp giấy có cả ảnh minh họa", 
                "query": """SELECT "size_type", "length", "tolerance", "image_path" 
                            FROM "packaging_sizes"
                            WHERE "id" = 'BT0KH00000KP0000'
                            AND "size_type" = 'Chiều rộng đáy bao'
                            LIMIT 5;"""
            },
        ]
        example_prompt = PromptTemplate.from_template("User input: {input} \nSQL query: {query}")
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(model="text-embedding-3-large"),
            FAISS,
            k=5,
            input_keys=["input"],
        )
        prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            # prefix="You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}"+str(db_info)+"\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
            prefix="You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run. Unless otherwise specificed, do not return more than {top_k} rows.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
            suffix="Message history: "+str(history)+"\nUser input: {input}\nSQL query: ",
            input_variables=["input", "top_k", "table_info"],
        )
        return prompt
    
    def ensemble_retriever(self):
        tree_retriever = self.tree_retriever()
        summarize_retriever = self.multivector_summary_retriever()
        ensemble_retriever = EnsembleRetriever(
            retrievers=[tree_retriever, summarize_retriever], weights=[0.7, 0.3]
        )
        return ensemble_retriever
    
    def tree_retriever(self):
        def treeTraversal(branches, caps):
            for branch in branches:
                if branch.metadata['subtree'] == []:
                    branch.metadata['img_cap'] = []
                    for cap in caps:
                        link, img_cap = cap.split("%")
                        if link in branch.page_content:
                            branch.metadata['img_cap'] += [img_cap]
                else:
                    branch.metadata['subtree'] = treeTraversal(branch.metadata['subtree'], caps)
            return branches
        patterns = [
            r'(?<=\n)Chương \d+: .+',  # Mục lớn như "Chương 1: TỔNG QUAN VỀ BAO BÌ"
            r'(?<=\n)\d+\.\d+ .+',  # Mục nhỏ hơn như "1.1 ĐẶC ĐIỂM CHUNG"
            r'(?<=\n)\d+\.\d+\.\d+ .+',  # Mục nhỏ hơn nữa như "1.1.1 Đặc điểm chung của bao bì"
            r'(?<=\n)\d+\.\d+\.\d+\.\d+ .+'  # Mục nhỏ hơn nữa như "1.1.2.1 Tính năng chứa đựng sản phẩm"
        ]
        file_names = os.listdir('doc')
        for file_name in file_names:
            with open("img_cap/"+file_name[:-4]+"_cap.txt", 'r', encoding='utf-8') as f:
                caps = f.readlines()
            loaders = [
                TextLoader("doc/"+file_name, encoding = 'UTF-8'),
            ]
            docs = []
            for loader in loaders:
                docs+=[loader.load()]
            splitter = PatternSplitter(patterns)
            print("before split")
            split_docs = splitter.split_documents(docs[0])
            print("after split")
            split_docs = treeTraversal(split_docs, caps)
            print("after go through")
            retriever = TreeRetriever(documents=split_docs)
        return retriever
    
    def multivector_subchunk_retriever(self):
        db_path = "./chroma_langchain_db"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print("Đã xóa thư mục Chroma cũ.")
        vectorstore = Chroma(
            collection_name="full_documents", embedding_function=self.embedding, persist_directory="./chroma_langchain_db"
        )
        store = InMemoryByteStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,  separators=["\n\n"])
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
        file_names = os.listdir('doc')
        for file_name in file_names:
            with open("img_cap/"+file_name[:-4]+"_cap.txt", 'r', encoding='utf-8') as f:
                caps = f.readlines()
            loaders = [
                TextLoader("doc/"+file_name, encoding = 'UTF-8'),
            ]
            docs = []
            for loader in loaders:
                docs.extend(loader.load())
            docs = text_splitter.split_documents(docs)
            doc_ids = [str(uuid.uuid4()) for _ in docs]
            sub_docs = []
            for i, doc in enumerate(docs):
                _id = doc_ids[i]
                _sub_docs = child_text_splitter.split_documents([doc])
                for _doc in _sub_docs:
                    _doc.metadata[id_key] = _id
                sub_docs.extend(_sub_docs)
                doc.metadata['img_cap'] = []
                for cap in caps:
                    link, img_cap = cap.split("%")
                    if link in doc.page_content:
                        doc.metadata['img_cap'] += [img_cap]
            retriever.vectorstore.add_documents(sub_docs)
            retriever.docstore.mset(list(zip(doc_ids, docs)))
        return retriever
    
    def multivector_summary_retriever(self):
        db_path = "./chroma_langchain_db"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print("Đã xóa thư mục Chroma cũ.")
        vectorstore = Chroma(
            collection_name="summaries", embedding_function=self.embedding, persist_directory="./chroma_langchain_db"
        )
        store = InMemoryByteStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,  separators=["\n\n"])
        chain = summary_chain()
        file_names = os.listdir('doc')
        for file_name in file_names:
            with open("img_cap/"+file_name[:-4]+"_cap.txt", 'r', encoding='utf-8') as f:
                caps = f.readlines()
            loaders = [
                TextLoader("doc/"+file_name, encoding = 'UTF-8'),
            ]
            docs = []
            for loader in loaders:
                docs.extend(loader.load())
            docs = text_splitter.split_documents(docs)
            doc_ids = [str(uuid.uuid4()) for _ in docs]
            summaries = chain.batch(docs, {"max_concurrency": 5})
            summary_docs = [
                Document(page_content=s, metadata={id_key: doc_ids[i]})
                for i, s in enumerate(summaries)
            ]
            for doc in docs:
                doc.metadata['img_cap'] = []
                for cap in caps:
                    link, img_cap = cap.split("%")
                    if link in doc.page_content:
                        doc.metadata['img_cap'] += [img_cap]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, docs)))
        return retriever    
    
    def docs_gen(self, question):
        en_question = vn_2_en(question)
        try:
            _filter = LLMChainFilter.from_llm(self.model)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=self.retriever
            )
            docs = compression_retriever.invoke(en_question)
            if docs == []: docs = [Document(page_content="Không có tài liệu liên quan đến câu hỏi này.", metadata={})]
        except Exception as e:
            print(e)
            docs = [Document(page_content="Không có tài liệu liên quan đến câu hỏi này.", metadata={})]
        return docs
    
    def query_gen(self, question, config):
        db = connect_database()
        def sql_gen(question):
            trimmer = self.llm_memory()
            memory_chain = (
                RunnablePassthrough.assign(messages=itemgetter("question") | trimmer)
                | (lambda output: ({
                    "question": output["messages"]
                }))
            )
            question = memory_chain.invoke(question)
            history = question['question'][:-1]
            question['question'] = str(question['question'][-1])
            prompt = self.sql_code_prompt_gen(history)
            write_query = create_sql_query_chain(self.model, db, prompt)
            for _ in range(5):
                sql_code = write_query.invoke(question)
                if ": " in sql_code:
                    sql_code = sql_code.strip().split(": ")[1]
                if "```sql" in sql_code:
                    sql_code = sql_code.strip()[7:-4]
                if 'SQLQuery' in sql_code:
                    return sql_code
                try:
                    db.run(sql_code)
                    break
                except ProgrammingError as e:
                    print("Code gen: \n")
                    print(sql_code)
                    print("\nCode SQL gen bị sai, đang tiến hành gen lại")
                except Exception as e:
                    print(e)
                    break
            return sql_code
        write_query  = RunnableLambda(sql_gen)
        execute_query = QuerySQLDataBaseTool(db=db)
        chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | (lambda output: (output.update({'output': output['result']}), output)[-1])
        )
        sql_gen_with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history_mongodb,
            input_messages_key="question", #Which key is user's input
        )
        code = sql_gen_with_message_history.invoke(
            {
                "question": question
            },
            config=config,)
        delete_two_most_recent_message(config['configurable']['session_id'])
        return code, db

    def invoke(self, question: str, session_id: str) -> str:
        config = {"configurable": {"session_id": session_id}}
        trimmer = self.llm_memory()
        img_caps = []
        img_paths = []
        result = {'capable': False}
        #routing
        db = connect_database()
        sql_info = db.get_table_info()
        response = routing(question,sql_info, config)
        print("RESPONSE: ",response)
        if response == 'True':
            #SQL
            prompt = self.sql_prompt_gen()
            class Classification(BaseModel):
                output: str = Field(description="An answer of the question")
                capable: bool = Field(description="Can LLm answer the question, True is Yes, False is No")
            llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0).with_structured_output(
                Classification
            )
            chain = (
                RunnablePassthrough.assign(messages=itemgetter("question") | trimmer)
                | (lambda output: (output.update({'question': output['messages']}), output)[-1])
                | prompt
                | llm
                | (lambda x: {"capable": x.capable, "output": x.output})
            )
            with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history_mongodb,
                input_messages_key="question", #Which key is user's input
            )
            code, db = self.query_gen(question,config)
            pattern = r"img\\\\produces_property\\\\[^\s]+\.png"
            matches = re.findall(pattern, str(code['result']))
            if matches:
                matches.reverse()
                for idx, match in enumerate(matches):
                    img_paths += [match]
                    cap = "Hình "+str(idx+1)
                    img_caps += [cap]
                    code['result'] = code['result'].replace(match,"Ảnh minh họa: " + cap)
            try:
                db.run(code['query'])
                print(code['query'])
                print(code['result'])
                print("///////////////////////////////////")
                result = with_message_history.invoke(
                    {
                        "question": [HumanMessage(content=question)],
                        "query": code['query'],
                        "result": code['result']
                    },
                    config=config,
                )
            except Exception as e:
                print(e)
                result = {'capable': False}
            print("result: ",result)
        if response == 'False' or (response == 'True' and result['capable'] == False):
            #DOCS
            prompt = self.docs_prompt_gen()
            chain = (
                RunnablePassthrough.assign(messages=itemgetter("question") | trimmer)
                | (lambda output: (output.update({'question': output['messages']}), output)[-1])
                | prompt
                | self.model
                | self.parser
            )
            with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history_mongodb,
                input_messages_key="question", #Which key is user's input
            )
            docs = self.docs_gen(question)
            if docs[0].page_content != 'Không có tài liệu liên quan đến câu hỏi này.':
                img_caps = img_caps_gen(docs)
            docs = "\n\n".join(doc.page_content for doc in docs)
            print(docs)
            print("/////////////////////////////////////////////")
            result = with_message_history.invoke(
                {
                    "question": [HumanMessage(content=question)],
                    "doc": docs
                },
                config=config,
            )
        else:
            result = result['output']
            docs = ''
        return result, img_caps, img_paths, docs
    
    def stream(self, question: str, session_id: str) -> str:
        config = {"configurable": {"session_id": session_id}}
        trimmer = self.llm_memory()
        prompt = self.prompt_gen()
        chain = (
            RunnablePassthrough.assign(messages=itemgetter("question") | trimmer)
            | (lambda output: (output.update({'question': output['messages']}), output)[-1])
            | prompt
            | self.model
            | self.parser
        )
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history_mongodb,
            input_messages_key="question", #Which key is user's input
        )
        docs = self.docs_gen(question)
        print(docs)
        print("/////////////////////////////////////////////")
        dict_input = {
            "question": [HumanMessage(content=question)],
            "doc": docs
        }
        return with_message_history, dict_input, config