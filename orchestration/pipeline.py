import yaml

from models.llm import LocalLLM
from models.embeddings import LocalEmbeddings

from ingestion.loader import load_documents
from ingestion.chunking import split_documents
from ingestion.vectorstore import FAISSVectorStore

from retrieval.hybrid import HybridRetriever
from retrieval.multihop import MultiHopRetriever
from retrieval.reranker import Reranker

from generation.generator import generate_answer
from generation.reflection import reflect

from monitoring.logger import RAGLogger


def build_pipeline():

    config = yaml.safe_load(
        open("config/settings.yaml")
    )

    logger = RAGLogger()

    llm = LocalLLM(config["llm_model"])
    embeddings = LocalEmbeddings(config["embedding_model"])

    docs = load_documents([
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    ])

    splits = split_documents(
        docs,
        config["chunk_size"],
        config["chunk_overlap"]
    )

    vectorstore = FAISSVectorStore(embeddings)
    vectorstore.build(splits)

    hybrid = HybridRetriever(splits, vectorstore)

    multihop = MultiHopRetriever(
        hybrid,
        llm
    )

    reranker = Reranker(config["reranker_model"])

    def pipeline(question):

        docs = multihop.retrieve(question)

        docs = reranker.rerank(
            question,
            docs,
            config["top_k"]
        )

        answer = generate_answer(
            question,
            docs,
            llm
        )

        context = "\n\n".join([
            d.page_content for d in docs
        ])

        final_answer = reflect(
            question,
            answer,
            context,
            llm
        )

        logger.log({
            "question": question,
            "answer": final_answer
        })

        return {
            "generation": final_answer
        }

    return pipeline