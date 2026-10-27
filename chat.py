from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import config

TEMPLATE = """
You are an expert in answering questions about a pizza restaurant based ONLY on the provided reviews.

Here are the relevant reviews from customers:
{reviews}

Question: {question}

Answer the question using ONLY the information from the reviews above. If the reviews don't contain enough information to answer the question, say so. Do not make up information.
"""


def build_chain():
    model = ChatGoogleGenerativeAI(model=config.CHAT_MODEL)
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    return prompt | model


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _sources_payload(retrieved_docs) -> list[dict]:
    return [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in retrieved_docs
    ]


def answer(question: str, retriever, chain) -> dict:
    retrieved_docs = retriever.invoke(question)
    reviews = "\n\n".join(doc.page_content for doc in retrieved_docs)

    response = chain.invoke({"reviews": reviews, "question": question})

    return {
        "answer": extract_text(response.content),
        "sources": _sources_payload(retrieved_docs),
    }


def stream_answer(question: str, retriever, chain):
    """Yields dict events: one "sources" event, then "token" events, then "done"."""
    retrieved_docs = retriever.invoke(question)
    reviews = "\n\n".join(doc.page_content for doc in retrieved_docs)

    yield {"type": "sources", "sources": _sources_payload(retrieved_docs)}

    for chunk in chain.stream({"reviews": reviews, "question": question}):
        text = extract_text(chunk.content)
        if text:
            yield {"type": "token", "text": text}

    yield {"type": "done"}
