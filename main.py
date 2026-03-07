from orchestration.pipeline import build_pipeline
import multiprocessing

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn", force=True)

    rag = build_pipeline()

    question = "What is retrieval augmented generation?"

    response = rag(question)

    print("\nAnswer:\n")
    print(response["generation"])