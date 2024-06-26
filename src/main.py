"""
Read local text + PDF docs and use local LLM to answer questions

ref = https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/benchmarks/phi-3-mini-4k-instruct.ipynb
"""

from cornsnake import util_print

from . import util_data, config

# built-in data reader - consumes text and PDF files
from llama_index.core import SimpleDirectoryReader


def print_section(title):
    util_print.print_section(title)


# logging
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# LLM
def messages_to_prompt(messages):
    prompt = ""
    system_found = False
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}<|end|>\n"
            system_found = True
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}<|end|>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}<|end|>\n"
        else:
            prompt += f"<|user|>\n{message.content}<|end|>\n"

    # trailing prompt
    prompt += "<|assistant|>\n"

    if not system_found:
        prompt = "<|system|>\nYou are a helpful AI assistant.<|end|>\n" + prompt

    return prompt


print_section(f"Loading LLM... [{config.MODEL}]")

from llama_index.llms.ollama import Ollama

llm = Ollama(model=config.MODEL, request_timeout=360.0, temperature=config.TEMPERATURE)

from llama_index.core import Settings, PromptHelper

from llama_index.embeddings.ollama import OllamaEmbedding

Settings.llm = llm
Settings.embed_model = OllamaEmbedding(model_name=config.EMBEDDING_MODEL)

# indexes - vector + summary
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)
import os
from . import util_data

have_new_data = util_data.check_if_new_data(config.DOCS_LOCATION)

VECTOR_PERSIST_DIR = "./storage_vector"
SUMMARY_PERSIST_DIR = "./storage_summary"

documents = None
vector_index = None
summary_index = None
if (
    have_new_data
    or not os.path.exists(VECTOR_PERSIST_DIR)
    or not os.path.exists(SUMMARY_PERSIST_DIR)
):
    # load the documents and create the index
    print_section(f"Reading documents from {config.DOCS_LOCATION}")
    documents = SimpleDirectoryReader(config.DOCS_LOCATION).load_data()

    if config.IS_VECTOR_ENABLED:
        print_section("Building vector index...")
        vector_index = VectorStoreIndex.from_documents(documents)
        # store it for later
        vector_index.storage_context.persist(persist_dir=VECTOR_PERSIST_DIR)

    if config.IS_SUMMARY_ENABLED:
        print_section("Building summary index...")
        summary_index = SummaryIndex.from_documents(documents)
        # store it for later
        summary_index.storage_context.persist(persist_dir=SUMMARY_PERSIST_DIR)
else:
    if config.IS_VECTOR_ENABLED:
        print_section("Loading existing vector index...")
        storage_context = StorageContext.from_defaults(persist_dir=VECTOR_PERSIST_DIR)
        vector_index = load_index_from_storage(storage_context)

    if config.IS_SUMMARY_ENABLED:
        print_section("Loading existing summary index...")
        storage_context = StorageContext.from_defaults(persist_dir=SUMMARY_PERSIST_DIR)
        summary_index = load_index_from_storage(storage_context)

# router query engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

vector_tool = None
if config.IS_VECTOR_ENABLED:
    vector_tool = QueryEngineTool(
        vector_index.as_query_engine(),
        metadata=ToolMetadata(
            name="vector_search",
            description="Useful for searching for specific facts.",
        ),
    )

summary_tool = None
if config.IS_SUMMARY_ENABLED:
    summary_tool = QueryEngineTool(
        summary_index.as_query_engine(response_mode="tree_summarize"),
        metadata=ToolMetadata(
            name="summary",
            description="Useful for summarizing an entire document.",
        ),
    )

# multiselector
from llama_index.core.query_engine import RouterQueryEngine

query_engine = None
if config.IS_SUMMARY_ENABLED and config.IS_VECTOR_ENABLED:
    query_engine = RouterQueryEngine.from_defaults(
        [vector_tool, summary_tool], select_multi=False
    )
elif config.IS_SUMMARY_ENABLED:
    query_engine = summary_tool.query_engine
elif config.IS_VECTOR_ENABLED:
    query_engine = vector_tool.query_engine
else:
    raise ValueError(
        "One of IS_SUMMARY_ENABLED or IS_VECTOR_ENABLED must be enabled - please check config.py"
    )


def _display_response(response):
    print(response)


print_section("Starting query loop...")

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
