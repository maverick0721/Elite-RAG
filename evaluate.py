from orchestration.pipeline import build_pipeline
from evaluation.benchmark_dataset import benchmark
from evaluation.evaluator import RAGEvaluator
from evaluation.report import summarize
import torch.distributed as dist
import multiprocessing

def main():
    multiprocessing.set_start_method("spawn", force=True)
    pipeline = build_pipeline()
    evaluator = RAGEvaluator(pipeline, benchmark)
    results = evaluator.run()
   
    summary = summarize(results)

    print("\nEvaluation Summary:\n")
    print(summary)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()