from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:

    def __init__(self, documents, dense_store):
        self.documents = documents
        self.dense_store = dense_store

        tokenized_docs = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query, k=5):
        dense_docs = self.dense_store.search(query, k)

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        top_sparse_idx = np.argsort(scores)[::-1][:k]
        sparse_docs = [self.documents[i] for i in top_sparse_idx]

        combined = list({
            doc.page_content: doc
            for doc in dense_docs + sparse_docs
        }.values())

        return combined[:k]