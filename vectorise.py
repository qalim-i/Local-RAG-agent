from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma 
from langchain_core.documents import Document 
import pandas as pd 
import os


df= pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="qwen3-embedding:4B")

db_location = "./chroma_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    print(f"[VECTORISING] Creating new vector store and adding {len(df)} documents...")
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
    )

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"[VECTORISED] Successfully added {len(documents)} documents to vector store")
else:
    print(f"[VECTORISOR] Loading existing vector store from {db_location}")

# Verify the collection has documents
collection_count = vector_store._collection.count()
print(f"[VECTORISE] Vector store contains {collection_count} documents")

retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )             