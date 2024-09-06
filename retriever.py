from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.retrievers.document_compressors import LLMListwiseRerank

class TreeRetriever(BaseRetriever):
    documents: List[Document]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        summaries = []
        for idx, doc in enumerate(self.documents):
            summaries += [Document(page_content=doc.metadata['summary'], metadata={"idx":idx})]
        llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
        compressor = LLMListwiseRerank.from_llm(llm, top_n=1)
        rank = compressor.compress_documents(summaries, query)
        rank_1 = self.documents[rank[0].metadata['idx']]
        if rank_1.metadata['subtree'] != []:
            retriever = TreeRetriever(documents=rank_1.metadata['subtree'])
            return retriever.invoke(query)
        else:
            result = self.documents[rank[0].metadata['idx']].copy()
            result.page_content = result.metadata['full_content']
            return result
            