import numpy as np
import re

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

class HybridRetriever:

    def __init__(self, documents, dense_store):
        self.documents = documents
        self.dense_store = dense_store

        tokenized_docs = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs) if BM25Okapi is not None else None

    def retrieve(self, query, k=5):
        dense_docs = self.dense_store.search(query, k)

        if self.bm25 is not None:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            top_sparse_idx = np.argsort(scores)[::-1][:k]
            sparse_docs = [self.documents[i] for i in top_sparse_idx]
        else:
            q_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2}

            def lexical_score(doc):
                d_terms = set(re.findall(r"[a-zA-Z0-9]+", doc.page_content.lower()))
                return len(q_terms.intersection(d_terms))

            sparse_docs = sorted(self.documents, key=lexical_score, reverse=True)[:k]

        combined = list({
            doc.page_content: doc
            for doc in dense_docs + sparse_docs
        }.values())

        return combined[:k]