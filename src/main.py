"""
Read local text + PDF docs and use phi3 LLM to answer questions

ref = https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/benchmarks/phi-3-mini-4k-instruct.ipynb
"""

from cornsnake import util_print

# built-in data reader - consumes text and PDF files
from llama_index.core import SimpleDirectoryReader

DOCS_LOCATION = "./data"


def print_section(title):
    util_print.print_section(title)


print_section(f"Reading documents from {DOCS_LOCATION}")

documents = SimpleDirectoryReader(DOCS_LOCATION).load_data()

# logging
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# LLM
from llama_index.llms.huggingface import HuggingFaceLLM


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


print_section("Loading LLM...")

llm = HuggingFaceLLM(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    model_kwargs={
        "trust_remote_code": True,
    },
    generate_kwargs={"do_sample": True, "temperature": 0.1},
    tokenizer_name="microsoft/Phi-3-mini-4k-instruct",
    query_wrapper_prompt=(
        "<|system|>\n"
        "You are a helpful AI assistant.<|end|>\n"
        "<|user|>\n"
        "{query_str}<|end|>\n"
        "<|assistant|>\n"
    ),
    messages_to_prompt=messages_to_prompt,
    is_chat_model=True,
)

from llama_index.core import Settings, PromptHelper
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# NOTE: try avoid error like "Token indices sequence length is longer than the specified maximum sequence length for this model"
CONTEXT_WINDOW = 4096
CHUNK_SIZE = 512
Settings.chunk_size = CHUNK_SIZE
Settings.context_window = CONTEXT_WINDOW
MAX_OUTPUT_TOKENS = 1024
prompt_helper = PromptHelper(
    context_window=CONTEXT_WINDOW, num_output=MAX_OUTPUT_TOKENS
)
Settings.prompt_helper = prompt_helper

# index - vector
from llama_index.core import VectorStoreIndex

print_section("Building vector index...")
vector_index = VectorStoreIndex.from_documents(documents)

# index - summary
from llama_index.core import SummaryIndex

print_section("Building summary index...")
summary_index = SummaryIndex.from_documents(documents)

# TODO - if index already exists, and it is newer than any data, then load it.
# TODO - save index - ref https://docs.llamaindex.ai/en/stable/getting_started/starter_example/

# router query engine

from llama_index.core.tools import QueryEngineTool, ToolMetadata

vector_tool = QueryEngineTool(
    vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts.",
    ),
)

summary_tool = QueryEngineTool(
    summary_index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document.",
    ),
)

# multiselector
from llama_index.core.query_engine import RouterQueryEngine

query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool],
    select_multi=True,
)


def _display_response(response):
    print(response)


print_section("Starting query loop...")

from tenacity import retry, wait_fixed, stop_after_attempt


@retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
def send_prompt(user_query):
    return query_engine.query(user_query)


while True:
    user_query = input("How can I help? [Press ENTER to exit] >>")
    if not user_query:
        print("Goodbye for now")
        break
    response = send_prompt(user_query)
    _display_response(response)
