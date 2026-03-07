import numpy as np

def summarize(results):

    avg_score = np.mean([
        r["semantic_similarity"]
        for r in results
    ])

    return {
        "avg_semantic_similarity": float(avg_score)
    }