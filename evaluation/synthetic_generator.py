def generate_synthetic_qa(documents, llm, num_samples=10):

    dataset = []

    for doc in documents[:num_samples]:

        prompt = f"""
Generate one question-answer pair from this text.

Text:
{doc.page_content}

Format:
Question:
Answer:
"""

        output = llm.generate(prompt)

        parts = output.split("Answer:")

        if len(parts) == 2:

            question = parts[0].replace(
                "Question:", ""
            ).strip()

            answer = parts[1].strip()

            dataset.append({
                "question": question,
                "ground_truth_answer": answer
            })

    return dataset