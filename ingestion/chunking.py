from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=400, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)