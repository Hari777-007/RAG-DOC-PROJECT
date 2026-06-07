from typing import List, Optional
from src.state.rag_state import ragstate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class node:
    """ defines the nodes for the RAG workflow"""
    def __init__(self,retriever,llm):
        self.retriever=retriever
        self.llm=llm
        self._agent=None

    def retrieve_docs(self,state:ragstate)->List[Document]:
        docs=self.retriever.invoke(state.question)
        return ragstate(question=state.question, retrieved_docs=docs)


    def build_tools(self):
        @tool
        def retriever_tool_fn(query: str) -> str:
            """Retrieve docs from vector store"""
            docs = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        wiki_api = WikipediaAPIWrapper()
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

        return [retriever_tool_fn, wiki_tool]    





    def build_agent(self):
        tools=self.build_tools()
        system_prompt = """
You are a ReAct agent.

You have access to tools:
- retriever_tool: use ONLY for user documents
- wikipedia: use ONLY for general knowledge


"""
       
        self._agent=create_react_agent(self.llm,tools=tools,prompt=system_prompt)


    def generate_answer(self,state:ragstate)->str:
        if self._agent is None:
            self.build_agent()

        result= self._agent.invoke({"messages":[HumanMessage(content=state.question)]})
        messages=result.get("messages",[])
        answer:Optional[str]=None
        if messages:
            answer_msg=messages[-1]
            answer=getattr(answer_msg,"content",None)
        return ragstate(question=state.question,retrieved_docs=state.retrieved_docs,answer=answer or 
        "Sorry, I couldn't generate an answer based on the provided information.")
            





        


