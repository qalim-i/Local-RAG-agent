from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import hashlib
import glob
import json
import shutil

import config


def _hash_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _coerce_metadata_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_documents(csv_files: list[str]) -> tuple[list[Document], list[str]]:
    documents = []
    ids = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"[INGEST] Reading {len(df)} rows from {csv_file}...")
        for i, row in df.iterrows():
            content = "\n".join(f"{col}: {value}" for col, value in row.items())
            metadata = {"source_file": csv_file}
            for col, value in row.items():
                coerced = _coerce_metadata_value(value)
                if coerced is not None:
                    metadata[col] = coerced
            doc_id = f"{csv_file}:{i}"
            documents.append(Document(page_content=content, metadata=metadata, id=doc_id))
            ids.append(doc_id)

    return documents, ids


def _current_manifest(csv_files: list[str]) -> dict:
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "files": {f: _hash_file(f) for f in csv_files},
    }


def _load_stored_manifest() -> dict | None:
    if not config.MANIFEST_PATH.exists():
        return None
    with open(config.MANIFEST_PATH, "r") as f:
        return json.load(f)


def build_store(force: bool = False) -> Chroma:
    csv_files = sorted(glob.glob(str(config.BASE_DIR / "*.csv")))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the project directory.")

    current_manifest = _current_manifest(csv_files)
    stored_manifest = _load_stored_manifest()
    needs_rebuild = force or stored_manifest != current_manifest

    if needs_rebuild and config.DB_DIR.exists():
        print("[INGEST] CSV files or embedding model changed, rebuilding vector store...")
        shutil.rmtree(config.DB_DIR)

    add_documents = not config.DB_DIR.exists()

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.DB_DIR),
        embedding_function=embeddings,
    )

    if add_documents:
        documents, ids = _load_documents(csv_files)
        vector_store.add_documents(documents=documents, ids=ids)
        with open(config.MANIFEST_PATH, "w") as f:
            json.dump(current_manifest, f)
        print(f"[INGEST] Added {len(documents)} documents to the vector store")
    else:
        print(f"[INGEST] Loaded existing vector store from {config.DB_DIR}")

    return vector_store


if __name__ == "__main__":
    store = build_store()
    print(f"[INGEST] Vector store contains {store._collection.count()} documents")
