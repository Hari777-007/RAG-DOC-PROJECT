import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
class config:
    """ configuration class for rag system"""

    groq_api_key:str=os.getenv("GROQ_API_KEY")
    llm_model="llama-3.3-70b-versatile"
    chunk_size=500
    chunk_overlap=50

    default_urls=[
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]



    @classmethod
    def get_llm(cls):
        """ initializes the language model based on the configuration"""
        os.environ["OPENAI_API_KEY"]=cls.groq_api_key
        return  init_chat_model("llama-3.3-70b-versatile",model_provider="groq")
