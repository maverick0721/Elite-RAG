import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class RetrieverDistillation:

    def __init__(self, student_model="BAAI/bge-small-en"):
        self.student = SentenceTransformer(student_model)

    def train(self, query_doc_pairs, teacher_scores, epochs=1):

        optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=2e-5
        )

        loss_fn = nn.MSELoss()

        for epoch in range(epochs):

            for (query, doc), score in zip(query_doc_pairs, teacher_scores):

                emb_q = self.student.encode(
                    [query],
                    convert_to_tensor=True
                )

                emb_d = self.student.encode(
                    [doc],
                    convert_to_tensor=True
                )

                sim = torch.cosine_similarity(emb_q, emb_d)

                loss = loss_fn(
                    sim,
                    torch.tensor([score]).to(sim.device)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()