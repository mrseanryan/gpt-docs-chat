# ref https://docs.llamaindex.ai/en/stable/examples/ingestion/ingestion_gdrive/

from cornsnake import util_print

# built-in data reader - consumes text and PDF files
from llama_index.core import SimpleDirectoryReader

from . import config

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.redis import RedisVectorStore

from redisvl.schema import IndexSchema

embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_BAAI)

util_print.print_section(f"Loading LLM... [{config.MODEL}]")

from llama_index.llms.ollama import Ollama

llm = Ollama(model=config.MODEL, request_timeout=360.0, temperature=config.TEMPERATURE)

from llama_index.core import Settings, PromptHelper

from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = llm
Settings.embed_model = OllamaEmbedding(model_name=config.EMBEDDING_MODEL)

custom_schema = IndexSchema.from_dict(
    {
        "index": {"name": "gdrive", "prefix": "doc"},
        # customize fields that are indexed
        "fields": [
            # required fields for llamaindex
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            # custom vector field for bge-small-en-v1.5 embeddings
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 384,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    }
)

util_print.print_section("Setting up vector store on redis...")

vector_store = RedisVectorStore(
    schema=custom_schema,
    redis_url="redis://localhost:6379",
)

# Optional: clear vector store if exists
# if vector_store.index_exists():
#     vector_store.delete_index()

# Set up the ingestion cache layer
cache = IngestionCache(
    cache=RedisCache.from_host_and_port("localhost", 6379),
    collection="redis_cache",
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        embed_model,
    ],
    docstore=RedisDocumentStore.from_host_and_port(
        "localhost", 6379, namespace="document_store"
    ),
    vector_store=vector_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

from llama_index.core import VectorStoreIndex

# TODO try SummaryIndex
vector_index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)

util_print.print_section(f"Reading documents from {config.DOCS_LOCATION}")
documents = SimpleDirectoryReader(config.DOCS_LOCATION).load_data()

nodes = pipeline.run(documents=documents)
print(f"Ingested {len(nodes)} Nodes")

query_engine = vector_index.as_query_engine()


def _display_response(response):
    print(response)


util_print.print_section("Starting query loop...")

from tenacity import retry, wait_fixed, stop_after_attempt


@retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
def send_prompt(user_query):
    return query_engine.query(user_query)


USER_EXIT = "bye"

while True:
    user_query = input(
        "How can I help? [to exit, type 'bye' and press ENTER] [for a summary, say 'summary'] >>"
    )
    if user_query.lower() == USER_EXIT.lower():
        print("Goodbye for now")
        break
    if not user_query:
        continue
    response = send_prompt(user_query)
    _display_response(response)
