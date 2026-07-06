import streamlit as st
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.config.config import config
from src.doc_ingestion.doc_process import documentprocessor
from src.vectorstore.vectorstore import VectorStore
from src.graphbuilder.graph_builder import graphbuilder


st.set_page_config(page_title="RAG Search 🤖",page_icon="🔍" ,layout="centered")


st.markdown(
"""
    <style>
    .stButton > button {
    width:100%;
    background-color: #4CAF50;
    color:white;
    font-weight: bold;
    }
    </style>
    """,unsafe_allow_html=True)


def init_session_state():
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []


@st.cache_resource
def initialize_rag():
    try:
        llm=config.get_llm()
        doc_processor=documentprocessor(chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap)
        vector_store=VectorStore()
        urls=config.default_urls
        documents=doc_processor.process_url(urls)
        vector_store.create_retriever(documents)

        graph_build=graphbuilder(retriever=vector_store.get_retriever(),llm=llm)

        graph_build.build()

        return graph_build,len(documents)

    except Exception as e:
        st.error(f"failed to initialize:{str(e)}")
        return None,0    

def main():
    init_session_state()

    st.title("🧐 RAG DOC SEARCH")
    st.markdown("Ask questios about the loaded documents")

    if not st.session_state.initialized:
        with st.spinner("loading system...."):
            rag_system,num_chunks=initialize_rag()
            if rag_system:
                st.session_state.rag_system=rag_system
                st.session_state.initialized=True
                st.success(f" ✅ Done! System ready! ({num_chunks} document chunks loaded)")

    st.markdown("---")

    with st.form("Search form"):
        question=st.text_input("Enter your question",
        placeholder="What woul you like to know?")
        submit=st.form_submit_button("🔍 Search")

    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("searching.."):
                start_time=time.time()
                result=st.session_state.rag_system.run(question)
                elapsed_time=time.time() - start_time


                st.session_state.history.append({
                    "question":question,
                    "answer":result["answer"],
                    "time":elapsed_time 
                })

                st.markdown("### 💡 Answer")
                st.success(result["answer"])

                with st.expander("Source Documents"):
                    for i,doc in enumerate(result["retrieved_docs"],1):
                        st.text_area(
                            f"document{i}",
                            doc.page_content[:300]+"....",
                            height=100,
                            disabled=True)

                st.caption(f"⌛ Response time {elapsed_time:.2f} seconds")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Recent Searches")

        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time:{item['time']:.2f}s")
                st.markdown("")

if __name__ =="__main__":
    main()                

























