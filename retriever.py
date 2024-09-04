from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.retrievers.document_compressors import FlashrankRerank

class TreeRetriever(BaseRetriever):
    documents: List[Document]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        matching_documents = []
        summaries = []
        exist_sub = False
        for doc in self.documents:
            if doc.metadata['subtree'] != []:
                exist_sub = True
            summaries += [Document(page_content=doc.metadata['summary'])]
        FlashrankRerank.update_forward_refs()
        compressor = FlashrankRerank()
        rank = compressor.compress_documents(summaries, query)
        print(rank)
            