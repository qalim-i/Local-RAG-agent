from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import pandas as pd
import glob
import json
import os
import shutil
import time

load_dotenv()

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_RATE_LIMIT = 100  # free-tier embed_content items per minute
db_location = "./chroma_db"
manifest_path = os.path.join(db_location, "source_manifest.json")

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

csv_files = sorted(glob.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in the current directory.")

current_manifest = {
    "embedding_model": EMBEDDING_MODEL,
    "files": {f: os.path.getmtime(f) for f in csv_files},
}

stored_manifest = None
if os.path.exists(manifest_path):
    with open(manifest_path, "r") as f:
        stored_manifest = json.load(f)

if stored_manifest != current_manifest and os.path.exists(db_location):
    print("[VECTORISE] CSV files or embedding model changed, rebuilding vector store...")
    shutil.rmtree(db_location)

add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"[VECTORISING] Reading {len(df)} rows from {csv_file}...")
        for i, row in df.iterrows():
            content = " ".join(str(value) for value in row.values)
            metadata = {"source": csv_file, **{col: str(value) for col, value in row.items()}}
            doc_id = f"{csv_file}:{i}"
            documents.append(Document(page_content=content, metadata=metadata, id=doc_id))
            ids.append(doc_id)

vector_store = Chroma(
    collection_name="csv_documents",
    persist_directory=db_location,
    embedding_function=embeddings
    )

if add_documents:
    for start in range(0, len(documents), EMBEDDING_RATE_LIMIT):
        chunk_docs = documents[start:start + EMBEDDING_RATE_LIMIT]
        chunk_ids = ids[start:start + EMBEDDING_RATE_LIMIT]
        vector_store.add_documents(documents=chunk_docs, ids=chunk_ids)
        if start + EMBEDDING_RATE_LIMIT < len(documents):
            print("[VECTORISING] Pausing to stay under the free-tier embedding rate limit...")
            time.sleep(61)
    with open(manifest_path, "w") as f:
        json.dump(current_manifest, f)
    print(f"[VECTORISED] Successfully added {len(documents)} documents to vector store")
else:
    print(f"[VECTORISOR] Loading existing vector store from {db_location}")

# Verify the collection has documents
collection_count = vector_store._collection.count()
print(f"[VECTORISE] Vector store contains {collection_count} documents")

retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
