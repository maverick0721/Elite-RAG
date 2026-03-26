import yaml

from models.llm import LocalLLM
from models.embeddings import LocalEmbeddings

from ingestion.loader import load_documents, load_inline_documents
from ingestion.chunking import split_documents
from ingestion.vectorstore import FAISSVectorStore

from retrieval.hybrid import HybridRetriever
from retrieval.multihop import MultiHopRetriever
from retrieval.reranker import Reranker

from generation.generator import generate_answer
from generation.reflection import reflect

from monitoring.logger import RAGLogger


DEFAULT_QUICKSTART_CORPUS = [
    (
        "Retrieval-augmented generation (RAG) combines document retrieval and language "
        "generation. It improves factual grounding by giving the generator relevant context."
    ),
    (
        "Hybrid retrieval combines dense embedding search with sparse lexical search such as BM25. "
        "This usually improves recall compared to either approach alone."
    ),
    (
        "A reranker scores query-document pairs and improves ranking quality. Reflection can check "
        "whether an answer is supported by context and reduce unsupported claims."
    ),
]


def _build_docs(config, quickstart: bool):
    if quickstart:
        corpus = config.get("quickstart_corpus", DEFAULT_QUICKSTART_CORPUS)
        return load_inline_documents(corpus)

    docs = load_documents(config.get("source_urls", []))
    if docs:
        return docs
    # If network/doc loading fails, still keep the app runnable.
    return load_inline_documents(DEFAULT_QUICKSTART_CORPUS)


def build_pipeline(config_path="config/settings.yaml", quickstart=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger = RAGLogger()
    device = config.get("device", "auto")

    llm_backend = config.get("llm_backend", "auto")
    embedding_backend = config.get("embedding_backend", "auto")
    reranker_backend = config.get("reranker_backend", "auto")

    if quickstart:
        llm_backend = "rule_based"
        embedding_backend = "tfidf"
        reranker_backend = "lexical"

    llm = LocalLLM(config["llm_model"], backend=llm_backend, device=device)
    embeddings = LocalEmbeddings(
        config["embedding_model"],
        backend=embedding_backend,
        device=device,
    )

    docs = _build_docs(config, quickstart=quickstart)

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

    reranker = Reranker(
        config["reranker_model"],
        backend=reranker_backend,
        device=device,
    )

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