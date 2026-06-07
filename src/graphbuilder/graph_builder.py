from langgraph.graph import StateGraph,END
from src.state.rag_state import ragstate
from src.nodes.reactnode import node

class graphbuilder:
    """ builds and manages the langgraph workflow"""
    def __init__(self,retriever,llm):
         self.nodes=node(retriever,llm)
         self.graph=None


    def build(self):

        builder=StateGraph(ragstate)
        builder.add_node("retriever",self.nodes.retrieve_docs)
        builder.add_node("responder",self.nodes.generate_answer)
        builder.set_entry_point("retriever") 
        builder.add_edge("retriever","responder")
        builder.add_edge("responder",END)
        self.graph=builder.compile()
        return self.graph

    def run(self,question):
        if self.graph is None:
            self.build()
        initial_state=ragstate(question=question)
        result=self.graph.invoke(initial_state)    
        return result




