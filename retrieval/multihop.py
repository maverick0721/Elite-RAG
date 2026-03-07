class MultiHopRetriever:

    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm

    def reformulate(self, question, context):

        prompt = f"""
        Given the question and partial context,
        generate a follow-up query needed to fully answer it.

        Question: {question}
        Context: {context}
        Follow-up Query:
        """

        return self.llm.generate(prompt)

    def retrieve(self, question, k=5):

        first_pass = self.base_retriever.retrieve(question, k)
        context = " ".join([doc.page_content for doc in first_pass])

        followup_query = self.reformulate(question, context)
        second_pass = self.base_retriever.retrieve(followup_query, k)

        return list({
            doc.page_content: doc
            for doc in first_pass + second_pass
        }.values())