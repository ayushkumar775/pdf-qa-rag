# 📚 Multi-PDF Q&A System (RAG)

A Retrieval-Augmented Generation (RAG) application that allows users to upload 
multiple PDF documents and ask questions across all of them simultaneously.

## Features
- Upload multiple PDFs at once
- Ask questions in natural language
- Get accurate, document-grounded answers
- View exact source (PDF name + page number) used for each answer

## Tech Stack
- **LangChain** - RAG orchestration framework
- **ChromaDB** - Vector database for storing embeddings
- **HuggingFace Embeddings** - Free embedding model (all-MiniLM-L6-v2)
- **Groq API (Llama 3.1)** - Fast, free LLM inference
- **Streamlit** - Web interface

## How It Works (RAG Pipeline)
1. **Load**: PDFs are loaded and text is extracted
2. **Chunk**: Text is split into smaller overlapping chunks
3. **Embed**: Each chunk is converted into vector embeddings
4. **Store**: Embeddings are stored in ChromaDB
5. **Retrieve**: User query is matched against stored chunks via similarity search
6. **Augment**: Relevant chunks are added to the LLM prompt as context
7. **Generate**: LLM generates final answer grounded in the retrieved content

## Setup Instructions

### 1. Clone/Download the project

### 2. Create virtual environment
\`\`\`bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
\`\`\`

### 3. Install dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Add your API key
Create a \`.env\` file and add:
\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`
Get free API key from: https://console.groq.com

### 5. Run the app
\`\`\`bash
streamlit run app.py
\`\`\`

## Project Structure
\`\`\`
pdf-qa-rag/
├── app.py
├── requirements.txt
├── utils/
│   ├── pdf_processor.py
│   ├── vector_store.py
│   └── qa_engine.py
└── README.md
\`\`\`