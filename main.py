from orchestration.pipeline import build_pipeline
import multiprocessing
import torch.distributed as dist


def main():
    multiprocessing.set_start_method("spawn", force=True)

    rag = build_pipeline()

    try:
        while True:
            question = input("\nAsk something (type 'exit' to quit): ")

            if question.lower() in ["exit", "quit", "q"]:
                break

            result = rag(question)

            print("\nAnswer:\n")
            print(result["generation"])

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()