from orchestration.pipeline import build_pipeline
from evaluation.benchmark_dataset import benchmark
from evaluation.evaluator import RAGEvaluator
from evaluation.report import summarize
import multiprocessing
import argparse

try:
    import torch.distributed as dist
except Exception:
    dist = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run Elite-RAG evaluation")
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Run evaluation with lightweight CPU-safe defaults",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    multiprocessing.set_start_method("spawn", force=True)
    pipeline = build_pipeline(quickstart=args.quickstart)
    evaluator = RAGEvaluator(pipeline, benchmark)
    results = evaluator.run()

    summary = summarize(results)

    print("\nEvaluation Summary:\n")
    print(summary)

    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()