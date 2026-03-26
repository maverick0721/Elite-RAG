from dataclasses import dataclass

try:
    from langchain_core.documents import Document
except Exception:
    @dataclass
    class Document:
        page_content: str


def load_documents(urls):
    try:
        from langchain_community.document_loaders import WebBaseLoader
    except Exception:
        return []

    docs = []
    for url in urls:
        try:
            docs.extend(WebBaseLoader(url).load())
        except Exception:
            # Keep pipeline resilient when a source is unavailable.
            continue
    return docs


def load_inline_documents(texts):
    return [Document(page_content=text) for text in texts]