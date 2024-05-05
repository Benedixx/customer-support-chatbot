import os
from pinecone import Pinecone
from dotenv import load_dotenv
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
import torch


class ChatBot:
    def __init__(self) -> None:
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        Settings.llm = HuggingFaceLLM(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            model_kwargs={
                "trust_remote_code": True,
                "torch_dtype": torch.float16
            },
            generate_kwargs={"do_sampel": True, "temperature": 0.1},
            tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
            query_wrapper_prompt=(
                "<|system|>\n"
                "You are a helpful AI assistant.<|end|>\n"
                "<|user|>\n"
                "{query_str}<|end|>\n"
                "<|assistant|>\n"
            ),
        )
        Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pc_index = self.pc.Index(host=os.getenv("PINECONE_HOST"))

    def request_response(self, query: str):
        vector_store = PineconeVectorStore(pinecone_index=self.pc_index)
        index = VectorStoreIndex.from_vector_store(vector_store)
        retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
        query_text =  f"<|user|>\n{query} <|end|>\n<|assistant|>"
        

        try:
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[LLMRerank(choice_batch_size=2, top_n=2)],
            )
            response = query_engine.query(query_text)
        except:
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
            )
            response = query_engine.query(query_text)

        return response


if __name__ == "__main__":
    chatbot = ChatBot()
    query = "What is available product for cat?"
    response = chatbot.request_response(query)
    print(response)
    pass
