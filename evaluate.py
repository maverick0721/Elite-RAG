from orchestration.pipeline import build_pipeline

from evaluation.benchmark_dataset import benchmark
from evaluation.evaluator import RAGEvaluator
from evaluation.report import summarize

pipeline = build_pipeline()

evaluator = RAGEvaluator(
    pipeline,
    benchmark
)

results = evaluator.run()

summary = summarize(results)

print("\nEvaluation Summary:\n")
print(summary)