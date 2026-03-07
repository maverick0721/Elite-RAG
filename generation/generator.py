def generate_answer(question, documents, llm):

    context = "\n\n".join([
        doc.page_content for doc in documents
    ])

    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer.

If the context does not contain the answer,
say "I don't know".

Question:
{question}

Context:
{context}

Answer:
"""

    return llm.generate(prompt)