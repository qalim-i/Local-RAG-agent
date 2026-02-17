from langchain_ollama.llms import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate 
from vectorise import retriever

model= OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant based ONLY on the provided reviews.

Here are the relevant reviews from customers:
{reviews}

Question: {question}

Answer the question using ONLY the information from the reviews above. If the reviews don't contain enough information to answer the question, say so. Do not make up information.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model 

while True: 
    print("\n" +"-"*70)
    question = input("Ask your question (Enter 'q' to quit): ")
    print("\n" +"-"*70)

    if question == "q":
        break

    retrieved_docs = retriever.invoke(question)
    reviews = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    print(f"\n[DEBUG] Retrieved {len(retrieved_docs)} reviews:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n  Review {i}: {doc.page_content[:100]}...")
    print("\n" + "="*70)
    
    response = chain.invoke({"reviews": reviews ,"question": question})
    print(response)
