DOCS_LOCATION = "./data"

EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL_BAAI = "BAAI/bge-small-en-v1.5"  # TODO is this more efficient than nomic-embed-text?

MODEL = "llama3"  # phi3 llama3
TEMPERATURE = 0.7

# For performance, try disabling one of these:
IS_SUMMARY_ENABLED = False
IS_VECTOR_ENABLED = True

CLEAR_OUT_INDEX = False  # normally False unless code has changed how storage (local or redis) is used

REDIS_SIMILARITY_TOP_K=10  # higher lists more result-docs, but may be slower
