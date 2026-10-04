"""
This module handles:
1. Creating embeddings
2. Storing them in ChromaDB (vector database)
3. Setting up retriever
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def get_embedding_model():
    """
    Returns a free HuggingFace embedding model.
    This converts text chunks into vector numbers.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def create_vector_store(chunks):
    """
    Takes text chunks, converts to embeddings, 
    and stores them in ChromaDB (in-memory).
    """
    embeddings = get_embedding_model()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    return vectorstore


def get_retriever(vectorstore, k=4):
    """
    Creates a retriever from vectorstore.
    k = number of relevant chunks to retrieve per query.
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    return retriever