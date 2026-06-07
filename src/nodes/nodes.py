from src.state.rag_state import ragstate


class ragnodes:

    def __init__(self, retriever, llm):
        self.retriever=retriever
        self.llm=llm


    def retrieve_docs(self, state:ragstate):
        docs=self.retriever.invoke(state.question)
        return ragstate(question=state.question,retrieved_docs=docs)

    def generate_answer(self, state:ragstate):
        context="\n\n".join([doc.page_content for doc in state.retrieved_docs])
        prompt=f"Answer the question based on the following retrieved documents:\n\n{context}\n\nQuestion: {state.question}\nAnswer:"
        response=self.llm.invoke(prompt)
        return ragstate(question=state.question,
        retrieved_docs=state.retrieved_docs,
        answer=response.content)
        s



