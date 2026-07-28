"""
embeddings.py

This module loads the embedding model
used to convert text into vector embeddings.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Load the Hugging Face embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
