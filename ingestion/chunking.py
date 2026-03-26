def split_documents(documents, chunk_size=400, chunk_overlap=50):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)
    except Exception:
        # Portable fallback splitter for quickstart/no-extra-deps environments.
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        for doc in documents:
            text = getattr(doc, "page_content", "")
            for i in range(0, len(text), step):
                piece = text[i : i + chunk_size]
                if piece:
                    chunks.append(type(doc)(page_content=piece))
        return chunks