from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from vectorise import retriever

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

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
    if isinstance(response.content, str):
        print(response.content)
    else:
        print("".join(block["text"] for block in response.content if block.get("type") == "text"))
