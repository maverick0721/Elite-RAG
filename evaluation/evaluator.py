from evaluation.metrics import semantic_similarity

class RAGEvaluator:

    def __init__(self, pipeline, dataset):
        self.pipeline = pipeline
        self.dataset = dataset

    def run(self):

        results = []

        for sample in self.dataset:

            output = self.pipeline(sample["question"])

            score = semantic_similarity(
                output["generation"],
                sample["ground_truth_answer"]
            )

            results.append({
                "question": sample["question"],
                "semantic_similarity": score
            })

        return results