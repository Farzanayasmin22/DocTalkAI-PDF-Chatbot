"""
rag_pipeline.py

This module handles the complete
Retrieval-Augmented Generation (RAG) workflow.
"""

def generate_answer(query, vector_store, llm, k=3):
    """
    Retrieve relevant document chunks and generate an answer using Gemini.
    """

    retrieved_docs = vector_store.similarity_search(query, k=k)

    if not retrieved_docs:
        return (
            "I couldn't find information related to that question in the uploaded PDF.\n\n"
            "Try:\n• Rephrasing your question\n• Asking about another topic\n• Requesting a summary of a section"
        ), []

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer the user's question.
Do not use any outside knowledge.

If the user asks for a summary:
- Summarize the document in clear, well-structured paragraphs.
- Include:
  • The main purpose of the document
  • The important concepts discussed
  • The key findings or contributions
  • The overall conclusion (if available)
- Do NOT list citations, references, or bibliography unless the user specifically asks about them.
- Keep the summary concise, informative, and easy to understand.

For all other questions:
- Answer clearly using complete sentences.
- Explain the answer in a simple and detailed manner based only on the provided context.

If the answer is not found in the context, reply exactly:
"I couldn't find that information in the uploaded document."

Context = {context}
Question = {query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = (
        response.content[0]["text"]
        if isinstance(response.content, list)
        else response.content
    )

    return answer, retrieved_docs
