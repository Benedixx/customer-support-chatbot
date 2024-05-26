from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore


con_string = "postgresql://postgres:nyanpasu@localhost:5432/nyanpasu"
db_name = "nyanpasu"
url = make_url(con_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    connection_string=con_string,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name="company",
    embed_dim=384,
)