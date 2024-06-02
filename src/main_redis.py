# ref https://docs.llamaindex.ai/en/stable/examples/ingestion/ingestion_gdrive/
# ref https://docs.llamaindex.ai/en/stable/examples/ingestion/redis_ingestion_pipeline/

import os

from cornsnake import util_print, util_color, util_input

from cornsnake import config as corsnake_config
corsnake_config.IS_INTERACTIVE = True

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
        "index": {"name": "redis_vector_store", "prefix": "doc"},
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
if config.CLEAR_OUT_INDEX and vector_store.index_exists():
    util_print.print_section("Clearing out index")
    vector_store.delete_index()

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
    vector_store=RedisVectorStore(
        schema=custom_schema,
        redis_url="redis://localhost:6379",
    ),
    cache=IngestionCache(
        cache=RedisCache.from_host_and_port("localhost", 6379),
        collection="redis_cache",
    ),
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_vector_store(
    pipeline.vector_store, embed_model=embed_model
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

util_print.print_section(f"Reading documents from {config.DOCS_LOCATION}")
documents = SimpleDirectoryReader(config.DOCS_LOCATION, filename_as_id=True).load_data()

nodes = pipeline.run(documents=documents)
print(f"Ingested {len(nodes)} Nodes")

query_engine = vector_index.as_query_engine(similarity_top_k=config.REDIS_SIMILARITY_TOP_K)

def _display_response_sources(source_nodes):
    for node in source_nodes:
        print(f" - [{node.metadata['file_path']}] [score: {round(node.score, 2)}] [{node.metadata['last_modified_date']}]")
    if len(source_nodes) == config.REDIS_SIMILARITY_TOP_K:
        util_print.print_with_color("note: there may be more documents available - please ask a more specific question, or edit config.py setting REDIS_SIMILARITY_TOP_K", util_color.CONFIG_COLOR)


def _display_response(response):
    print(response)
    if response.source_nodes:
        _display_response_sources(response.source_nodes)


util_print.print_section("Starting query loop...")

from tenacity import retry, wait_fixed, stop_after_attempt


@retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
def send_prompt(user_query):
    return query_engine.query(user_query)

USER_EXIT = "bye"

while True:
    user_query = util_input.input_required("How can I help? [to exit, type 'bye' and press ENTER] [for a summary, say 'summary' or 'What documents do you see?'] >>", "")
    if user_query.lower() == USER_EXIT.lower():
        print("Goodbye for now")
        break
    if not user_query:
        continue
    response = send_prompt(user_query)
    _display_response(response)
