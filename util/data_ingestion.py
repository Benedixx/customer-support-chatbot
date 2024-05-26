import os
from sqlalchemy import make_url
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    SimpleDirectoryReader,
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

if __name__ == "__main__":
    Settings.node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=10)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

    dir_reader = SimpleDirectoryReader(
        input_dir="company_data", file_extractor={"pdf": PDFReader()}
    )

    documents = dir_reader.load_data()
    print(f"Loaded {len(documents)} documents")

    for document in documents:
        document.metadata["last_accessed_date"] = "26/05/2024"

    con_string = "postgresql://postgres:nyanpasu@localhost:5432/nyanpasu"
    db_name = "nyanpasu"
    url = make_url(con_string)
    pg_store = PGVectorStore.from_params(
        database=db_name,
        host=url.host,
        connection_string=con_string,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name="company",
        embed_dim=384,
    )
    storage_context = StorageContext.from_defaults(vector_store=pg_store)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        show_progress=True,
        storage_context=storage_context,
    )
