"""
This module handles:
1. Setting up the LLM (Groq)
2. Creating the RAG chain (Retrieval + Generation)
"""

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os


def get_llm():
    """
    Initializes and returns the Groq LLM.
    Using openai/gpt-oss-20b as recommended by Groq (Aug 2026).
    """
    llm = ChatGroq(
        model="openai/gpt-oss-120b",    # ✅ Updated model (llama-3.1-8b-instant deprecated)
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )
    return llm


def create_qa_chain(retriever):
    llm = get_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain


def get_answer(qa_chain, query):
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    source_documents = result["source_documents"]
    return answer, source_documents