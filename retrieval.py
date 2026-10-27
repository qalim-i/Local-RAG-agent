import config
from ingest import build_store


def get_retriever(k: int | None = None):
    vector_store = build_store()
    return vector_store.as_retriever(search_kwargs={"k": k or config.TOP_K})
