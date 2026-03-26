from sklearn.metrics.pairwise import cosine_similarity

_model = None
_vectorizer = None

try:
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer

    _vectorizer = TfidfVectorizer()


def semantic_similarity(answer, ground_truth):
    if _model is not None:
        emb1 = _model.encode([answer])
        emb2 = _model.encode([ground_truth])
        return float(cosine_similarity(emb1, emb2)[0][0])

    matrix = _vectorizer.fit_transform([answer, ground_truth]).toarray()
    return float(cosine_similarity([matrix[0]], [matrix[1]])[0][0])