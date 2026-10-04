"""
This module handles:
1. Loading PDF files
2. Splitting them into chunks
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   
import tempfile
import os


def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name
    return temp_path


def load_pdf(file_path, source_name):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata["source_file"] = source_name
    
    return documents


def process_multiple_pdfs(uploaded_files):
    all_documents = []
    
    for uploaded_file in uploaded_files:
        temp_path = save_uploaded_file(uploaded_file)
        documents = load_pdf(temp_path, uploaded_file.name)
        all_documents.extend(documents)
        os.unlink(temp_path)
    
    return all_documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks