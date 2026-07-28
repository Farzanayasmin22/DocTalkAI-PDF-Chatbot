"""
vector_store.py

This module creates the ChromaDB
vector database for semantic search.
"""

from langchain_chroma import Chroma


def create_vector_store(chunks, embedding_model, persist_directory):
    """
    Store document embeddings in ChromaDB.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
