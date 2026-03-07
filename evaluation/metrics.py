from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(answer, ground_truth):

    emb1 = model.encode([answer])
    emb2 = model.encode([ground_truth])

    return cosine_similarity(emb1, emb2)[0][0]