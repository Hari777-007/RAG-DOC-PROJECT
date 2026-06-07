from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List,Union
from pathlib import Path
from langchain_community.document_loaders import (WebBaseLoader,PyPDFLoader,TextLoader,PyPDFDirectoryLoader)

class documentprocessor:
    """ handle doc loading and processing  """
    def __init__(self,chunk_size:int=500,chunk_overlap:int=50):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)


    def load_from_url(self,url:str) -> List[Document]:
        """ load doc from url and split into chunks """
        loader=WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self,dir_path:Union[str,path]) -> List[Document]:
        """ load all pdfs from a directory and split into chunks """
        loader=PyPDFDirectoryLoader(str(dir_path))
        return loader.load()

    def load_from_txt(self,file_path:Union[str,path]) -> List[Document]:
        """ load doc from txt file and split into chunks """
        loader=TextLoader(str(file_path),encoding='utf-8')
        return loader.load()
    def load_from_pdf(self,file_path:Union[str,path]) -> List[Document]:
        """ load doc from pdf file and split into chunks """
        loader=PyPDFLoader(str("data"))
        return loader.load()


    def load_documents(self,sources:List[str])->List[Document]:
        """ load docs from multiple sources and split into chunks """
        docs:List[Document]=[]
        for src in sources:
            if src.startswith("https://") or src.startswith("http://"):
                docs.extend(self.load_from_url(src))
            path = Path("data") 
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError (f"Unsupported file type: {src}. "
                                    "Supported types: URL, PDF directory, TXT file."
                
                )
            return docs
    def split_documents(self,documents:List[Document])->List[Document]:
        """ split documents into chunks """
        return self.text_splitter.split_documents(documents)


    def process_url(self,urls:List[str])->List[Document]:
        """ process docs from urls """
        docs=self.load_documents(urls)
        return self.split_documents(docs)


                                        
