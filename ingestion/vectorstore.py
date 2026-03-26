import numpy as np

try:
    import faiss
except Exception:
    faiss = None

class FAISSVectorStore:

    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.index = None
        self.documents = []
        self.embeddings_matrix = None

    def build(self, documents):
        self.documents = documents

        texts = [doc.page_content for doc in documents]
        if hasattr(self.embeddings_model, "fit"):
            self.embeddings_model.fit(texts)

        embeddings = self.embeddings_model.embed(
            texts
        )
        self.embeddings_matrix = np.array(embeddings).astype("float32")

        if faiss is not None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings_matrix)

    def search(self, query, k=5):
        query_emb = self.embeddings_model.embed([query])
        query_vec = np.array(query_emb).astype("float32")
        if self.index is not None:
            scores, indices = self.index.search(query_vec, k)
            return [self.documents[i] for i in indices[0]]

        # Portable fallback: cosine-ish ranking via dot product.
        scores = np.dot(self.embeddings_matrix, query_vec[0])
        indices = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in indices]