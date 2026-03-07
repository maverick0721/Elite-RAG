from langchain_community.document_loaders import WebBaseLoader

def load_documents(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]