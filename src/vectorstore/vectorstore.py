from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """ manages the vector store application """
    def __init__(self):
        self.embedding=OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = None
        self.retriever = None


    def create_retriever(self,documents:List[Document]):
        self.vectorstore=FAISS.from_documents(documents,self.embedding)
        self.retriever=self.vectorstore.as_retriever()

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("vector store not initiated yet")
        return self.retriever

    def retrieve(self,query:str,k:int=4) -> List[Document]:
        if self.retriever is None:
            raise ValueError("vector store not initiated yet")
        return self.retriever.invoke(query)        


