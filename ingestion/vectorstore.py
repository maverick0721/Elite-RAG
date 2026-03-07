import faiss
import numpy as np

class FAISSVectorStore:

    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.index = None
        self.documents = []

    def build(self, documents):
        self.documents = documents

        embeddings = self.embeddings_model.embed(
            [doc.page_content for doc in documents]
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query, k=5):
        query_emb = self.embeddings_model.embed([query])
        scores, indices = self.index.search(
            np.array(query_emb).astype("float32"),
            k
        )

        return [self.documents[i] for i in indices[0]]