import numpy as np


def _get_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class LocalEmbeddings:
    def __init__(self, model_name: str, backend: str = "auto", device: str = "auto"):
        self.backend = backend
        self.device = _get_device(device)
        self.model = None
        self.vectorizer = None

        if backend in {"auto", "sentence_transformers"}:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name, device=self.device)
                self.backend = "sentence_transformers"
                return
            except Exception:
                if backend == "sentence_transformers":
                    raise

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer()
        self.backend = "tfidf"

    def fit(self, texts):
        if self.backend == "tfidf":
            self.vectorizer.fit(texts)

    def embed(self, texts):
        if self.backend == "sentence_transformers":
            return self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(self.vectorizer.transform(texts).toarray(), dtype="float32")