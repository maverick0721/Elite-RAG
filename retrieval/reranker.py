import re


def _get_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Reranker:
    def __init__(self, model_name, backend: str = "auto", device: str = "auto"):
        self.backend = backend
        self.model = None
        resolved_device = _get_device(device)

        if backend in {"auto", "cross_encoder"}:
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(model_name, device=resolved_device)
                self.backend = "cross_encoder"
                return
            except Exception:
                if backend == "cross_encoder":
                    raise

        self.backend = "lexical"

    def rerank(self, query, documents, top_k=5):
        if self.backend == "cross_encoder" and self.model is not None:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.model.predict(pairs)
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked[:top_k]]

        q_terms = {t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2}

        def lexical_score(doc):
            d_terms = set(re.findall(r"[a-zA-Z0-9]+", doc.page_content.lower()))
            return len(q_terms.intersection(d_terms))

        ranked = sorted(documents, key=lexical_score, reverse=True)
        return ranked[:top_k]