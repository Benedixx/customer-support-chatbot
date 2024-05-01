import os
from pinecone import Pinecone
from dotenv import load_dotenv
from llama_index.readers.file import PDFReader
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from datetime import datetime
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc_index = pc.Index(host=os.getenv("PINECONE_HOST"))
hf_token = os.getenv("HF_TOKEN")

Settings.llm = HuggingFaceLLM(model_name = "microsoft/Phi-3-mini-4k-instruct",
                              model_kwargs={
                                  "trust_remote_code": True,
                              },
                              generate_kwargs={"do_sampel": True, "temperature": 0.1},
                              tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
                              query_wrapper_prompt=(
                                "<|system|>\n"
                                "You are a helpful AI assistant.<|end|>\n"
                                "<|user|>\n"
                                "{query_str}<|end|>\n"
                                "<|assistant|>\n"
                                ))
Settings.embed_model = HuggingFaceEmbedding(
    "BAAI/bge-small-en-v1.5"
)
Settings.node_parser = SimpleNodeParser(chunk_size=300, chunk_overlap=10)

if __name__ == "__main__":
    dir_reader = SimpleDirectoryReader(input_dir="company_data", file_extractor={"pdf": PDFReader()})
    
    documents = dir_reader.load_data()
    print(f"Loaded {len(documents)} documents")
    
    for document in documents :
        document.metadata["last_accessed_date"] = datetime.now()
    
    vector_store = PineconeVectorStore(pinecone_index=pc_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(documents,
                                            show_progress=True,
                                            storage_context=storage_context)
    