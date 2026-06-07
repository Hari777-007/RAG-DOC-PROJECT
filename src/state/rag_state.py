from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class ragstate(BaseModel):
    """ manages the state of the RAG application """
    question:str
    retrieved_docs:List[Document]=[]
    answer:str=""