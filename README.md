# DocuWise AI

DocuWise AI is an AI-powered legal document assistant designed to help users extract insights and answer questions from any PDFs. It utilizes Retrieval-Augmented Generation (RAG) with a locally hosted fine-tuned language model to provide accurate and context-aware answers without relying on external APIs.

## Features

- Multi-document support with individual vector stores
- Context-aware question answering using RAG pipeline
- Local embedding and retrieval using FAISS
- Streamlit-based UI for easy interaction
- Document source displayed with each answer
- Full reset option to clear all uploaded documents

## Technology Stack

- **Language Model**: deepset/roberta-base-squad2 (locally hosted)
- **Embeddings**: HuggingFace sentence transformers
- **Vector Database**: FAISS
- **Document Loading**: PDFPlumber via LangChain
- **Frameworks**: LangChain, Hugging Face Transformers, Streamlit

## Example Workflow

1. Upload any PDF
2. Select the document from the dropdown
3. Ask a question like:
   - What does Article 5 say about termination?
   - Are there any confidentiality clauses?
4. The system retrieves the top 3 relevant chunks and generates an answer based on them

## Notes

- All documents are embedded and stored locally using FAISS
- Supports multiple documents via isolated vector stores per file
- Uses `deepset/roberta-base-squad2` for offline reasoning, but can be swapped with any other Seq2Seq model
- No third-party APIs or internet access required after setup

## Acknowledgements

- Hugging Face
- LangChain
- FAISS
- Streamlit
