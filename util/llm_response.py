import os
from sqlalchemy import make_url
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from vector_db import vector_store
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.query_engine import RetrieverQueryEngine
import warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0")


class LLMResponse :
    def __init__(self) -> None:
        load_dotenv()
        Settings.node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=10)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))
        
    def generate_response(self, prompt:str, similarity_top_k:int = 3) :
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
        postprocessors = [LongContextReorder()]
        query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=postprocessors
        )
        result = query_engine.query(prompt)
        
        return result
    
if __name__ == "__main__" :
    llm = LLMResponse()
    response = llm.generate_response("what is the procedure of refund from nyanpasu store?")
    print(response)
    pass


