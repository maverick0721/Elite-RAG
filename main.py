from orchestration.pipeline import build_pipeline
import multiprocessing
import argparse

try:
    import torch.distributed as dist
except Exception:
    dist = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run Elite-RAG interactive app")
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Run with lightweight CPU-safe defaults and inline corpus",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Ask a single question and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    multiprocessing.set_start_method("spawn", force=True)

    rag = build_pipeline(quickstart=args.quickstart)

    try:
        if args.question:
            result = rag(args.question)
            print(result["generation"])
            return

        while True:
            question = input("\nAsk something (type 'exit' to quit): ")

            if question.lower() in ["exit", "quit", "q"]:
                break

            result = rag(question)

            print("\nAnswer:\n")
            print(result["generation"])

    finally:
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()