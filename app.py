"""
Main Streamlit App: Multi-PDF Q&A System using RAG
"""

import streamlit as st
from dotenv import load_dotenv
from utils.pdf_processor import process_multiple_pdfs, split_documents
from utils.vector_store import create_vector_store, get_retriever
from utils.qa_engine import create_qa_chain, get_answer

# Load environment variables (API keys)
load_dotenv()

# Page configuration
st.set_page_config(page_title="Multi-PDF Q&A (RAG)", page_icon="📚")

st.title("📚 Multi-PDF Question Answering System")
st.write("Upload multiple PDFs and ask questions across all of them!")

# ---------------------------
# STEP 1: File Upload (Multiple Files)
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True  # This enables MULTIPLE PDF upload
)

# ---------------------------
# STEP 2: Process PDFs (only runs when files are uploaded)
# ---------------------------
if uploaded_files:
    
    # Use session_state to avoid reprocessing on every interaction
    if "qa_chain" not in st.session_state or st.session_state.get("num_files") != len(uploaded_files):
        
        with st.spinner("Processing PDFs... This may take a moment."):
            
            # Load and combine all PDFs
            documents = process_multiple_pdfs(uploaded_files)
            
            # Split into chunks
            chunks = split_documents(documents)
            
            # Create vector store (embeddings)
            vectorstore = create_vector_store(chunks)
            
            # Create retriever
            retriever = get_retriever(vectorstore, k=4)
            
            # Create RAG chain
            qa_chain = create_qa_chain(retriever)
            
            # Store in session state so it persists across queries
            st.session_state.qa_chain = qa_chain
            st.session_state.num_files = len(uploaded_files)
            st.session_state.file_names = [f.name for f in uploaded_files]
        
        st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s): {', '.join(st.session_state.file_names)}")
    
    else:
        st.info(f"Using previously processed files: {', '.join(st.session_state.file_names)}")

    # ---------------------------
    # STEP 3: Question Input
    # ---------------------------
    st.divider()
    query = st.text_input("💬 Ask a question about your PDFs:")
    
    if query:
        with st.spinner("Searching documents and generating answer..."):
            answer, sources = get_answer(st.session_state.qa_chain, query)
        
        # ---------------------------
        # STEP 4: Display Answer
        # ---------------------------
        st.subheader("📝 Answer:")
        st.markdown(answer) 
        
        # ---------------------------
        # STEP 5: Display Sources (which PDF + chunk was used)
        # ---------------------------
        with st.expander("🔍 View Source Documents Used"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}** — From: `{doc.metadata.get('source_file', 'Unknown')}` (Page {doc.metadata.get('page', 'N/A')})")
                st.write(doc.page_content)
                st.divider()

else:
    st.info("👆 Please upload one or more PDF files to get started.")