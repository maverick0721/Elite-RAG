from sentence_transformers import SentenceTransformer

class LocalEmbeddings:

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cuda")

    def embed(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )