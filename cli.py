from chat import build_chain, answer
from retrieval import get_retriever


def main():
    retriever = get_retriever()
    chain = build_chain()

    while True:
        print("\n" + "-" * 70)
        question = input("Ask your question (Enter 'q' to quit): ").strip()
        print("\n" + "-" * 70)

        if question.lower() == "q":
            break
        if not question:
            continue

        result = answer(question, retriever, chain)

        print(f"\n[DEBUG] Retrieved {len(result['sources'])} reviews:")
        for i, source in enumerate(result["sources"], 1):
            print(f"\n  Review {i}: {source['content'][:100]}...")
        print("\n" + "=" * 70)

        print(result["answer"])


if __name__ == "__main__":
    main()
