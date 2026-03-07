from orchestration.pipeline import build_pipeline

rag = build_pipeline()

if __name__ == "__main__":

    question = "What is retrieval augmented generation?"

    response = rag(question)

    print("\nAnswer:\n")
    print(response["generation"])